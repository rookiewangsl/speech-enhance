"""Capture microphone audio, enhance it with RNNoise, and print live ASR text.

The front end runs continuously in the capture thread.  Completed VAD
segments are handed to a separate ASR worker, so slow decoding cannot make the
microphone callback miss audio frames.  This is endpointed ASR: a final
transcript appears after a stretch of silence, not token-by-token streaming.
"""

from __future__ import annotations

import argparse
import json
import queue
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from speech_frontend.audio import AudioData, write_audio
from speech_frontend.asr.timeline import render_transcript_timeline
from speech_frontend.rnnoise import RNNoiseLibrary, StreamingRNNoise48k
from speech_frontend.rnnoise.resampler import StreamingDownsampler3
from speech_frontend.vad.endpoint import (
    EndpointConfig,
    EndpointSegment,
    StreamingEndpointDetector,
)

INPUT_SAMPLE_RATE = 48_000
INPUT_FRAME_SAMPLES = 480
ASR_SAMPLE_RATE = 16_000
ASR_FRAME_SAMPLES = 160
ASR_FRAME_SECONDS = ASR_FRAME_SAMPLES / ASR_SAMPLE_RATE


def parse_device(value: str) -> int | str:
    """Accept either a numeric device index or a device-name substring."""

    try:
        return int(value)
    except ValueError:
        return value


def load_sounddevice():
    try:
        import sounddevice as sd
    except ImportError as error:
        raise RuntimeError(
            "Microphone demo requires the optional demo dependency. "
            "Run: ./.venv/bin/pip install -e '.[demo]'"
        ) from error
    return sd


def available_input_devices(sd) -> list[tuple[int, Any]]:
    """Return only devices that PortAudio reports as microphone-capable."""

    return [
        (index, device)
        for index, device in enumerate(sd.query_devices())
        if int(device["max_input_channels"]) > 0
    ]


def select_input_device(
    sd,
    requested_device: int | str | None,
    *,
    input_fn=input,
    output_fn=print,
) -> tuple[int | str, Any]:
    """Validate a supplied microphone or prompt for one in an interactive shell."""

    if requested_device is not None:
        try:
            return requested_device, sd.query_devices(requested_device, "input")
        except ValueError as error:
            raise ValueError(
                f"Not an input device: {requested_device!r}. "
                "Omit --device to choose interactively or use --list-devices."
            ) from error

    choices = available_input_devices(sd)
    if not choices:
        raise RuntimeError("No input devices are available")
    output_fn("Available input devices:")
    for index, device in choices:
        output_fn(
            f"  [{index}] {device['name']} "
            f"({device['max_input_channels']} input channel(s))"
        )
    valid_indices = {index for index, _ in choices}
    while True:
        try:
            response = input_fn("Choose an input device number: ").strip()
        except EOFError as error:
            raise RuntimeError(
                "No terminal input is available. Pass --device explicitly."
            ) from error
        try:
            selected = int(response)
        except ValueError:
            output_fn("Please enter one of the displayed numeric device indices.")
            continue
        if selected in valid_indices:
            return selected, sd.query_devices(selected, "input")
        output_fn("That device is not an available input device. Try again.")


def select_output_device(
    sd,
    requested_device: int | str | None,
) -> tuple[int | str | None, Any]:
    """Resolve the requested playback device, or use the system default."""

    try:
        return requested_device, sd.query_devices(requested_device, "output")
    except ValueError as error:
        device_name = "system default" if requested_device is None else repr(requested_device)
        raise ValueError(
            f"Not an output device: {device_name}. Use --list-devices "
            "or pass --output-device explicitly."
        ) from error


def load_asr_config(path: Path) -> dict[str, Any]:
    """Load only the fields required by the live Whisper demo."""

    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(f"ASR config does not exist: {path}") from None
    except json.JSONDecodeError as error:
        raise ValueError(f"ASR config is not valid JSON: {path}") from error
    if not isinstance(config, dict):
        raise ValueError("ASR config must be a JSON object")
    model = config.get("model")
    decoding = config.get("decoding")
    thresholds = config.get("thresholds")
    if not isinstance(model, dict) or not isinstance(decoding, dict):
        raise ValueError("ASR config needs object-valued model and decoding fields")
    if thresholds is not None and not isinstance(thresholds, dict):
        raise ValueError("ASR config thresholds must be an object when present")
    if not isinstance(model.get("name"), str) or not model["name"].strip():
        raise ValueError("ASR config model.name must be a non-empty string")
    if model.get("sample_rate_hz") != ASR_SAMPLE_RATE:
        raise ValueError("live ASR expects a 16 kHz ASR model configuration")
    return config


def whisper_options(config: dict[str, Any]) -> dict[str, Any]:
    """Translate the reusable ASR config into quiet Whisper call options."""

    options = dict(config["decoding"])
    options.update(config.get("thresholds", {}))
    # The worker emits one concise, application-controlled line per segment.
    options["verbose"] = False
    return options


class WhisperTranscriber:
    """Minimal local Whisper wrapper isolated from microphone capture."""

    def __init__(
        self,
        config: dict[str, Any],
        model_root: Path,
        device: str,
    ) -> None:
        try:
            import torch
            import whisper
        except ImportError as error:
            raise RuntimeError(
                "Live ASR requires the optional ASR dependency. "
                "Run: ./.venv-asr/bin/pip install -e '.[asr]'"
            ) from error
        if device == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested for ASR but is not available")
        self.model_name = str(config["model"]["name"])
        self.device = device
        self._torch = torch
        self.options = whisper_options(config)
        model_root.mkdir(parents=True, exist_ok=True)
        self._model = whisper.load_model(
            self.model_name,
            device=device,
            download_root=str(model_root),
        )

    def transcribe(self, samples: np.ndarray) -> str:
        result = self._model.transcribe(samples, **self.options)
        return str(result.get("text", "")).strip()

    def synchronize(self) -> None:
        """Make MPS timing include all queued accelerator work."""

        if self.device == "mps":
            self._torch.mps.synchronize()


@dataclass(frozen=True)
class SegmentJob:
    """A completed enhanced 16 kHz utterance ready for ASR."""

    sequence: int
    samples: np.ndarray
    speech_frames: int
    total_frames: int
    endpoint_reason: str
    endpoint_time: float
    start_seconds: float
    end_seconds: float


@dataclass(frozen=True)
class Transcript:
    """Transcript and latency metadata emitted by the ASR worker."""

    sequence: int
    text: str
    duration_seconds: float
    speech_frames: int
    total_frames: int
    endpoint_reason: str
    asr_seconds: float
    queue_seconds: float
    end_to_text_seconds: float
    start_seconds: float
    end_seconds: float


class ASRWorker(threading.Thread):
    """Decode endpointed utterances while capture keeps running."""

    def __init__(
        self,
        transcriber: WhisperTranscriber,
        jobs: queue.Queue[SegmentJob | None],
    ) -> None:
        super().__init__(name="live-asr", daemon=True)
        self.transcriber = transcriber
        self.jobs = jobs
        self.transcripts: list[Transcript] = []
        self.errors: list[str] = []

    def run(self) -> None:
        while True:
            job = self.jobs.get()
            try:
                if job is None:
                    return
                self.transcriber.synchronize()
                started = time.perf_counter()
                text = self.transcriber.transcribe(job.samples)
                self.transcriber.synchronize()
                finished = time.perf_counter()
                transcript = Transcript(
                    sequence=job.sequence,
                    text=text,
                    duration_seconds=job.samples.size / ASR_SAMPLE_RATE,
                    speech_frames=job.speech_frames,
                    total_frames=job.total_frames,
                    endpoint_reason=job.endpoint_reason,
                    asr_seconds=finished - started,
                    queue_seconds=started - job.endpoint_time,
                    end_to_text_seconds=finished - job.endpoint_time,
                    start_seconds=job.start_seconds,
                    end_seconds=job.end_seconds,
                )
                self.transcripts.append(transcript)
                printable_text = text if text else "(no speech recognized)"
                print(
                    f"[ASR {job.sequence:03d}] {printable_text} "
                    f"(segment={transcript.duration_seconds:.2f}s, "
                    f"ASR={transcript.asr_seconds:.2f}s)",
                    flush=True,
                )
            except BaseException as error:
                message = (
                    f"ASR segment {job.sequence if job is not None else '?'} failed: "
                    f"{type(error).__name__}: {error}"
                )
                self.errors.append(message)
                print(message, flush=True)
            finally:
                self.jobs.task_done()


def milliseconds_to_frames(milliseconds: float, name: str) -> int:
    if milliseconds <= 0.0:
        raise ValueError(f"{name} must be positive")
    return int(np.ceil(milliseconds / 10.0))


def make_endpoint_config(arguments: argparse.Namespace) -> EndpointConfig:
    return EndpointConfig(
        threshold_on=arguments.vad_on,
        threshold_off=arguments.vad_off,
        onset_frames=milliseconds_to_frames(arguments.onset_ms, "onset-ms"),
        hangover_frames=milliseconds_to_frames(
            arguments.hangover_ms,
            "hangover-ms",
        ),
        pre_roll_frames=milliseconds_to_frames(
            arguments.pre_roll_ms,
            "pre-roll-ms",
        ),
        minimum_speech_frames=milliseconds_to_frames(
            arguments.minimum_speech_ms,
            "minimum-speech-ms",
        ),
        maximum_segment_frames=milliseconds_to_frames(
            arguments.maximum_segment_ms,
            "maximum-segment-ms",
        ),
    )


def enqueue_segment(
    segment: EndpointSegment,
    jobs: queue.Queue[SegmentJob | None],
    sequence: int,
) -> bool:
    job = SegmentJob(
        sequence=sequence,
        samples=segment.samples,
        speech_frames=segment.speech_frames,
        total_frames=segment.total_frames,
        endpoint_reason=segment.reason,
        endpoint_time=time.perf_counter(),
        start_seconds=segment.start_frame * ASR_FRAME_SECONDS,
        end_seconds=segment.end_frame * ASR_FRAME_SECONDS,
    )
    try:
        jobs.put_nowait(job)
    except queue.Full:
        print(
            f"[ASR {sequence:03d}] dropped: pending ASR queue is full",
            flush=True,
        )
        return False
    print(
        f"[ASR {sequence:03d}] speech endpoint: "
        f"{job.samples.size / ASR_SAMPLE_RATE:.2f}s ({job.endpoint_reason})",
        flush=True,
    )
    return True


def process_enhanced_frames(
    enhanced_48k: np.ndarray,
    vad_probabilities: np.ndarray,
    downsampler: StreamingDownsampler3,
    endpoint: StreamingEndpointDetector,
) -> list[EndpointSegment]:
    """Downsample output and apply one RNNoise VAD score to each 10 ms frame."""

    if enhanced_48k.size == 0:
        if vad_probabilities.size:
            raise RuntimeError("RNNoise returned VAD scores without output audio")
        return []
    enhanced_16k = downsampler.process_chunk(enhanced_48k)
    expected_samples = vad_probabilities.size * ASR_FRAME_SAMPLES
    if enhanced_16k.size != expected_samples:
        raise RuntimeError(
            "RNNoise/VAD frame mismatch after 48 kHz to 16 kHz conversion: "
            f"{enhanced_16k.size} samples for {vad_probabilities.size} VAD frames"
        )
    completed: list[EndpointSegment] = []
    for index, probability in enumerate(vad_probabilities):
        start = index * ASR_FRAME_SAMPLES
        result = endpoint.process_frame(
            enhanced_16k[start : start + ASR_FRAME_SAMPLES],
            float(probability),
        )
        if result is not None:
            completed.append(result)
    return completed


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--duration",
        type=float,
        default=30.0,
        help="capture duration in seconds; use 0 to stop manually with Ctrl-C",
    )
    parser.add_argument(
        "--device",
        type=parse_device,
        help="input device index/name; omit to choose interactively",
    )
    parser.add_argument(
        "--output-device",
        type=parse_device,
        help="playback output device index/name; defaults to the system output",
    )
    parser.add_argument("--library", type=Path)
    parser.add_argument("--list-devices", action="store_true")
    parser.add_argument(
        "--asr-config",
        type=Path,
        default=Path("configs/asr_whisper_small_en.json"),
    )
    parser.add_argument(
        "--model-root",
        type=Path,
        default=Path(
            "/Volumes/T7/ProjectData/realtime_speech_enhancement/models/whisper"
        ),
    )
    parser.add_argument("--asr-device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--vad-on", type=float, default=0.60)
    parser.add_argument("--vad-off", type=float, default=0.35)
    parser.add_argument("--onset-ms", type=float, default=20.0)
    parser.add_argument("--hangover-ms", type=float, default=700.0)
    parser.add_argument("--pre-roll-ms", type=float, default=100.0)
    parser.add_argument("--minimum-speech-ms", type=float, default=200.0)
    parser.add_argument("--maximum-segment-ms", type=float, default=10_000.0)
    parser.add_argument("--max-pending-segments", type=int, default=4)
    arguments = parser.parse_args()
    if arguments.duration < 0.0:
        raise ValueError("duration must be zero or positive")
    if arguments.max_pending_segments <= 0:
        raise ValueError("max-pending-segments must be positive")
    return arguments


def main() -> None:
    arguments = parse_arguments()
    sd = load_sounddevice()
    if arguments.list_devices:
        print(sd.query_devices())
        return

    selected_device, input_device = select_input_device(sd, arguments.device)
    input_device_name = str(input_device["name"])
    selected_output_device, output_device = select_output_device(
        sd,
        arguments.output_device,
    )
    output_device_name = str(output_device["name"])
    endpoint_config = make_endpoint_config(arguments)
    asr_config = load_asr_config(arguments.asr_config)
    print(
        f"Loading Whisper {asr_config['model']['name']} on {arguments.asr_device}…",
        flush=True,
    )
    transcriber = WhisperTranscriber(
        asr_config,
        arguments.model_root,
        arguments.asr_device,
    )
    jobs: queue.Queue[SegmentJob | None] = queue.Queue(
        maxsize=arguments.max_pending_segments
    )
    worker = ASRWorker(transcriber, jobs)
    worker.start()

    audio_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=64)
    playback_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=128)
    callback_status: list[str] = []
    playback_callback_status: list[str] = []
    dropped_blocks = 0
    playback_dropped_blocks = 0
    playback_underflow_blocks = 0

    def callback(indata, frames, _time_info, status) -> None:
        nonlocal dropped_blocks
        if status:
            callback_status.append(str(status))
        if frames != INPUT_FRAME_SAMPLES:
            callback_status.append(f"unexpected block size: {frames}")
        try:
            audio_queue.put_nowait(indata[:, 0].copy())
        except queue.Full:
            dropped_blocks += 1

    def enqueue_playback(samples: np.ndarray) -> None:
        nonlocal playback_dropped_blocks
        if samples.size == 0:
            return
        if samples.ndim != 1 or samples.size % INPUT_FRAME_SAMPLES:
            raise RuntimeError("RNNoise playback samples must contain whole 10 ms frames")
        for start in range(0, samples.size, INPUT_FRAME_SAMPLES):
            try:
                playback_queue.put_nowait(
                    samples[start : start + INPUT_FRAME_SAMPLES].copy()
                )
            except queue.Full:
                playback_dropped_blocks += 1

    def playback_callback(outdata, frames, _time_info, status) -> None:
        nonlocal playback_underflow_blocks
        if status:
            playback_callback_status.append(str(status))
        if frames != INPUT_FRAME_SAMPLES:
            playback_callback_status.append(f"unexpected block size: {frames}")
        try:
            block = playback_queue.get_nowait()
        except queue.Empty:
            outdata.fill(0.0)
            playback_underflow_blocks += 1
            return
        if block.size != frames:
            playback_callback_status.append(
                f"playback block size mismatch: {block.size} != {frames}"
            )
            outdata.fill(0.0)
            playback_underflow_blocks += 1
            return
        outdata[:, 0] = block

    library = RNNoiseLibrary(arguments.library)
    stream = StreamingRNNoise48k(library)
    downsampler = StreamingDownsampler3()
    endpoint = StreamingEndpointDetector(endpoint_config)
    captured: list[np.ndarray] = []
    enhanced: list[np.ndarray] = []
    processing_seconds = 0.0
    queued_segments = 0
    dropped_segments = 0
    next_sequence = 1
    started = time.perf_counter()
    interrupted = False

    # Give the main processing loop a small head start.  Together with the
    # 10 ms RNNoise look-ahead this avoids an audible callback underflow at
    # stream startup while keeping monitoring latency below 50 ms.
    enqueue_playback(np.zeros(INPUT_FRAME_SAMPLES * 3, dtype=np.float32))

    def dispatch(completed: list[EndpointSegment]) -> None:
        nonlocal next_sequence, queued_segments, dropped_segments
        for segment in completed:
            if enqueue_segment(segment, jobs, next_sequence):
                queued_segments += 1
            else:
                dropped_segments += 1
            next_sequence += 1

    try:
        print(
            "Listening. Speak normally; a final transcript is printed after "
            "the configured silence interval.",
            flush=True,
        )
        print(
            f"Live RNNoise monitoring is playing through {output_device_name}. "
            "Use headphones to prevent speaker echo from feeding the microphone.",
            flush=True,
        )
        with sd.InputStream(
            samplerate=INPUT_SAMPLE_RATE,
            blocksize=INPUT_FRAME_SAMPLES,
            channels=1,
            dtype="float32",
            device=selected_device,
            callback=callback,
            latency="low",
        ), sd.OutputStream(
            samplerate=INPUT_SAMPLE_RATE,
            blocksize=INPUT_FRAME_SAMPLES,
            channels=1,
            dtype="float32",
            device=selected_output_device,
            callback=playback_callback,
            latency="low",
        ):
            try:
                while arguments.duration == 0.0 or (
                    time.perf_counter() - started < arguments.duration
                ):
                    try:
                        block = audio_queue.get(timeout=2.0)
                    except queue.Empty as error:
                        raise RuntimeError(
                            "No microphone frames arrived within 2 seconds. "
                            "Check the selected input device and macOS microphone "
                            "permission."
                        ) from error
                    captured.append(block)
                    processing_started = time.perf_counter()
                    result = stream.process_chunk(block)
                    processing_seconds += time.perf_counter() - processing_started
                    enhanced.append(result.samples)
                    enqueue_playback(result.samples)
                    dispatch(
                        process_enhanced_frames(
                            result.samples,
                            result.vad_probabilities,
                            downsampler,
                            endpoint,
                        )
                    )
            except KeyboardInterrupt:
                interrupted = True
                print("Stopping microphone capture…", flush=True)
            finally:
                tail = stream.flush()
                enhanced.append(tail.samples)
                enqueue_playback(tail.samples)
                # Tail samples exist only to preserve a saved waveform with exact input
                # length.  RNNoise does not provide matching VAD scores for this tail.
                downsampler.process_chunk(tail.samples)
                final_segment = endpoint.flush()
                if final_segment is not None:
                    dispatch([final_segment])

                # The OutputStream is still active here, so it can play the
                # delayed RNNoise tail before both streams are closed.
                drain_deadline = time.perf_counter() + 1.0
                while (
                    not playback_queue.empty()
                    and time.perf_counter() < drain_deadline
                ):
                    time.sleep(0.01)
    finally:
        stream.close()

    jobs.join()
    jobs.put(None)
    jobs.join()
    worker.join()

    if not captured:
        raise RuntimeError("microphone demo captured no audio")
    input_audio = np.concatenate(captured)
    output_audio = np.concatenate(enhanced)
    if output_audio.shape != input_audio.shape:
        raise RuntimeError("live ASR demo input/output length mismatch")

    output_root = arguments.output_dir
    output_root.mkdir(parents=True, exist_ok=True)
    write_audio(
        output_root / "microphone_raw.wav",
        AudioData(input_audio.astype(np.float32), INPUT_SAMPLE_RATE),
    )
    write_audio(
        output_root / "microphone_rnnoise.wav",
        AudioData(output_audio.astype(np.float32), INPUT_SAMPLE_RATE),
    )
    transcript_rows = [asdict(item) for item in worker.transcripts]
    (output_root / "transcripts.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in transcript_rows),
        encoding="utf-8",
    )
    timeline_path = output_root / "transcripts_timeline.png"
    timeline_error: str | None = None
    try:
        render_transcript_timeline(
            transcript_rows,
            timeline_path,
            title="ASR transcript timeline",
        )
    except RuntimeError as error:
        timeline_error = str(error)
        print(f"ASR timeline was not rendered: {timeline_error}", flush=True)
    captured_seconds = input_audio.size / INPUT_SAMPLE_RATE
    report = {
        "input_device": input_device_name,
        "output_device": output_device_name,
        "sample_rate": INPUT_SAMPLE_RATE,
        "block_samples": INPUT_FRAME_SAMPLES,
        "asr_sample_rate": ASR_SAMPLE_RATE,
        "requested_duration_seconds": arguments.duration,
        "captured_duration_seconds": captured_seconds,
        "stopped_by_keyboard_interrupt": interrupted,
        "wall_time_seconds": time.perf_counter() - started,
        "processing_rtf": processing_seconds / captured_seconds,
        "algorithmic_delay_samples": stream.algorithmic_delay_samples,
        "algorithmic_delay_ms": (
            1_000 * stream.algorithmic_delay_samples / INPUT_SAMPLE_RATE
        ),
        "asr_model": transcriber.model_name,
        "asr_device": transcriber.device,
        "endpoint": asdict(endpoint_config),
        "queued_segments": queued_segments,
        "dropped_segments": dropped_segments,
        "completed_transcripts": len(worker.transcripts),
        "transcript_timeline_png": str(timeline_path) if timeline_error is None else None,
        "transcript_timeline_error": timeline_error,
        "asr_errors": worker.errors,
        "input_peak": float(np.max(np.abs(input_audio), initial=0.0)),
        "output_peak": float(np.max(np.abs(output_audio), initial=0.0)),
        "input_clipping_samples": int(
            np.count_nonzero(np.abs(input_audio) >= 1.0)
        ),
        "output_clipping_samples": int(
            np.count_nonzero(np.abs(output_audio) >= 1.0)
        ),
        "dropped_blocks": dropped_blocks,
        "callback_status": callback_status,
        "playback_dropped_blocks": playback_dropped_blocks,
        "playback_underflow_blocks": playback_underflow_blocks,
        "playback_callback_status": playback_callback_status,
    }
    (output_root / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if worker.errors:
        raise RuntimeError("one or more live ASR segments failed")


if __name__ == "__main__":
    main()

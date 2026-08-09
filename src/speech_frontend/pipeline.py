"""Offline reference pipelines assembled from causal frame-level modules."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from speech_frontend.enhancement.spectral_subtraction import spectral_subtraction
from speech_frontend.enhancement.om_lsa import OMLSA, OMLSAConfig
from speech_frontend.enhancement.wiener import (
    DecisionDirectedWiener,
    DualUncertaintyWiener,
    WienerConfig,
    instantaneous_wiener_gain,
)
from speech_frontend.noise.mcra import MCRA, MCRAConfig
from speech_frontend.noise.imcra import IMCRA, IMCRAConfig
from speech_frontend.stft import STFT, STFTConfig, STFTResult
from speech_frontend.vad.statistical import StatisticalVAD, StatisticalVADConfig


@dataclass(frozen=True)
class EnhancementDiagnostics:
    """Per-frame traces needed for plots and ablations."""

    noise_psd: NDArray[np.float64]
    speech_presence_probability: NDArray[np.float64]
    gain: NDArray[np.float64]
    gain_speech_probability: NDArray[np.float64] | None


@dataclass(frozen=True)
class EnhancementResult:
    """Enhanced samples and the traces that created them."""

    samples: NDArray[np.float64]
    diagnostics: EnhancementDiagnostics


class ClassicalEnhancer:
    """A0/A1/A3/A4 reference implementations for paired evaluation."""

    def __init__(
        self,
        *,
        stft_config: STFTConfig | None = None,
        mcra_config: MCRAConfig | None = None,
        imcra_config: IMCRAConfig | None = None,
        wiener_config: WienerConfig | None = None,
        statistical_vad_config: StatisticalVADConfig | None = None,
    ) -> None:
        self.stft = STFT(stft_config)
        self.mcra_config = mcra_config or MCRAConfig()
        self.imcra_config = imcra_config or IMCRAConfig()
        self.wiener_config = wiener_config or WienerConfig()
        self.statistical_vad_config = statistical_vad_config

    def enhance(
        self,
        samples: NDArray[np.floating],
        *,
        method: str,
    ) -> EnhancementResult:
        """Enhance one waveform with a named, reproducible baseline."""

        if method not in {
            "identity",
            "spectral_subtraction",
            "mcra_instantaneous_wiener",
            "mcra_dd_wiener",
            "mcra_om_lsa",
            "imcra_om_lsa",
            "dual_uncertainty_wiener",
        }:
            raise ValueError(f"unknown enhancement method: {method}")
        analysis = self.stft.analyze(samples)
        spectra = analysis.spectrum
        noise_psd, p_n = MCRA(self.mcra_config).process_spectra(spectra)
        gain_speech_probability: NDArray[np.float64] | None = None

        if method == "identity":
            gains = np.ones_like(noise_psd)
            enhanced_spectra = spectra
        elif method == "spectral_subtraction":
            enhanced_spectra = spectral_subtraction(
                spectra,
                noise_psd,
                gain_floor=self.wiener_config.gain_floor,
            )
            gains = self._magnitude_gain(
                enhanced_spectra,
                spectra,
                gain_floor=self.wiener_config.gain_floor,
            )
        elif method == "mcra_instantaneous_wiener":
            gains = instantaneous_wiener_gain(
                np.abs(spectra) ** 2,
                noise_psd,
                gain_floor=self.wiener_config.gain_floor,
                maximum_snr=self.wiener_config.maximum_snr,
                epsilon=self.wiener_config.epsilon,
            )
            enhanced_spectra = gains * spectra
        else:
            if method == "mcra_dd_wiener":
                processor = DecisionDirectedWiener(self.wiener_config)
                enhanced_frames: list[NDArray[np.complexfloating]] = []
                gains_list: list[NDArray[np.float64]] = []
                for spectrum, frame_noise in zip(spectra, noise_psd, strict=True):
                    enhanced, gain = processor.process_frame(spectrum, frame_noise)
                    enhanced_frames.append(enhanced)
                    gains_list.append(gain)
                enhanced_spectra = np.stack(enhanced_frames)
                gains = np.stack(gains_list)
            elif method in {"mcra_om_lsa", "imcra_om_lsa"}:
                processor = OMLSA(
                    OMLSAConfig(
                        alpha_dd=self.wiener_config.alpha_dd,
                        gain_floor=self.wiener_config.gain_floor,
                        speech_absence_prior_min=(
                            self.wiener_config.speech_absence_prior_min
                        ),
                        speech_absence_prior_max=(
                            self.wiener_config.speech_absence_prior_max
                        ),
                        maximum_snr=self.wiener_config.maximum_snr,
                        epsilon=self.wiener_config.epsilon,
                    )
                )
                enhanced_frames = []
                gains_list = []
                p_g_list = []
                if method == "imcra_om_lsa":
                    noise_estimator = IMCRA(self.imcra_config)
                    noise_list = []
                    presence_list = []
                    previous_term = None
                    for spectrum in spectra:
                        frame_noise, frame_presence = noise_estimator.process_frame(
                            spectrum,
                            previous_decision_directed_term=previous_term,
                        )
                        enhanced, gain, p_g = processor.process_frame(
                            spectrum,
                            frame_noise,
                            local_speech_presence_probability=frame_presence,
                        )
                        enhanced_frames.append(enhanced)
                        gains_list.append(gain)
                        p_g_list.append(p_g)
                        noise_list.append(frame_noise)
                        presence_list.append(frame_presence)
                        posterior = np.abs(spectrum) ** 2 / np.maximum(
                            frame_noise,
                            self.wiener_config.epsilon,
                        )
                        previous_term = gain**2 * posterior
                    noise_psd = np.stack(noise_list)
                    p_n = np.stack(presence_list)
                else:
                    for spectrum, frame_noise, frame_presence in zip(
                        spectra,
                        noise_psd,
                        p_n,
                        strict=True,
                    ):
                        enhanced, gain, p_g = processor.process_frame(
                            spectrum,
                            frame_noise,
                            local_speech_presence_probability=frame_presence,
                        )
                        enhanced_frames.append(enhanced)
                        gains_list.append(gain)
                        p_g_list.append(p_g)
                enhanced_spectra = np.stack(enhanced_frames)
                gains = np.stack(gains_list)
                gain_speech_probability = np.stack(p_g_list)
            else:
                vad_probability = self._enhancement_vad_probability(samples, analysis)
                processor = DualUncertaintyWiener(self.wiener_config)
                enhanced_frames = []
                gains_list = []
                p_g_list = []
                for spectrum, frame_noise, frame_probability in zip(
                    spectra,
                    noise_psd,
                    vad_probability,
                    strict=True,
                ):
                    enhanced, gain, p_g = processor.process_frame(
                        spectrum,
                        frame_noise,
                        vad_speech_probability=float(frame_probability),
                    )
                    enhanced_frames.append(enhanced)
                    gains_list.append(gain)
                    p_g_list.append(p_g)
                enhanced_spectra = np.stack(enhanced_frames)
                gains = np.stack(gains_list)
                gain_speech_probability = np.stack(p_g_list)

        enhanced = self.stft.synthesize(
            STFTResult(
                spectrum=enhanced_spectra,
                original_length=analysis.original_length,
                left_padding=analysis.left_padding,
            )
        )
        return EnhancementResult(
            samples=enhanced,
            diagnostics=EnhancementDiagnostics(
                noise_psd=noise_psd,
                speech_presence_probability=p_n,
                gain=gains,
                gain_speech_probability=gain_speech_probability,
            ),
        )

    @staticmethod
    def _magnitude_gain(
        enhanced: NDArray[np.complexfloating],
        original: NDArray[np.complexfloating],
        *,
        gain_floor: float,
    ) -> NDArray[np.float64]:
        gain = np.divide(
            np.abs(enhanced),
            np.abs(original),
            out=np.ones_like(np.abs(original), dtype=np.float64),
            where=np.abs(original) > 1e-12,
        )
        return np.clip(gain, gain_floor, 1.0)

    def _enhancement_vad_probability(
        self,
        samples: NDArray[np.floating],
        analysis: STFTResult,
    ) -> NDArray[np.float64]:
        vad = StatisticalVAD(self.statistical_vad_config)
        result = vad.detect(samples)
        vad_centers = (
            np.arange(result.speech_probability.size)
            * vad.config.hop_length
            + vad.config.frame_length / 2.0
        )
        enhancement_centers = (
            np.arange(analysis.spectrum.shape[0]) * self.stft.config.hop_length
            - analysis.left_padding
            + self.stft.config.frame_length / 2.0
        )
        return np.interp(
            enhancement_centers,
            vad_centers,
            result.speech_probability,
            left=result.speech_probability[0],
            right=result.speech_probability[-1],
        )

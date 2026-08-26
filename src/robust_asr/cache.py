"""Content-addressed identities for ASR inference caches."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from robust_asr.config import canonical_sha256


@dataclass(frozen=True)
class CacheIdentity:
    """All identities that may change a robust-ASR hypothesis."""

    audio_sha256: str
    frontend_config_sha256: str
    model_revision: str
    adapter_sha256: str | None
    decoder_config_sha256: str
    normalizer_sha256: str

    def __post_init__(self) -> None:
        digest_fields = (
            "audio_sha256",
            "frontend_config_sha256",
            "decoder_config_sha256",
            "normalizer_sha256",
        )
        for field in digest_fields:
            value = getattr(self, field)
            if len(value) != 64 or any(
                character not in "0123456789abcdef" for character in value
            ):
                raise ValueError(f"{field} must be a lowercase SHA-256 digest")
        if self.adapter_sha256 is not None:
            value = self.adapter_sha256
            if len(value) != 64 or any(
                character not in "0123456789abcdef" for character in value
            ):
                raise ValueError(
                    "adapter_sha256 must be a lowercase SHA-256 digest"
                )
        if not self.model_revision:
            raise ValueError("model_revision cannot be empty")

    @property
    def digest(self) -> str:
        return canonical_sha256(asdict(self))


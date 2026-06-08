from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

QWEN_8B_MODEL = "Qwen/Qwen3-Embedding-8B"
SUPPORTED_LOCAL_HUGGINGFACE_PROVIDERS = {"local_huggingface"}


class EmbeddingBackend(Protocol):
    @property
    def name(self) -> str:
        """Embedding profile name."""

    @property
    def provider(self) -> str:
        """Embedding provider identifier."""

    @property
    def model(self) -> str:
        """Embedding model name."""

    @property
    def dimension(self) -> int | str:
        """Configured embedding dimension."""

    def build_llamaindex_embedding(self) -> Any:
        """Build a LlamaIndex-compatible embedding model."""


@dataclass(frozen=True)
class LocalHuggingFaceEmbeddingBackend:
    name: str
    config: dict[str, Any]

    @property
    def provider(self) -> str:
        return str(self.config["provider"])

    @property
    def model(self) -> str:
        return str(self.config["model"])

    @property
    def dimension(self) -> int | str:
        return self.config.get("dimension", "default")

    def build_llamaindex_embedding(self) -> Any:
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding

        kwargs: dict[str, Any] = {
            "model_name": self.model,
            "normalize": bool(self.config.get("normalize", True)),
        }

        optional_args = {
            "query_instruction": self.config.get("query_instruction"),
            "text_instruction": self.config.get(
                "text_instruction",
                self.config.get("instruction"),
            ),
            "max_length": self.config.get("max_length"),
            "embed_batch_size": self.config.get("embed_batch_size"),
            "cache_folder": self.config.get("cache_folder"),
            "trust_remote_code": self.config.get("trust_remote_code"),
            "device": self.config.get("device"),
        }
        kwargs.update({key: value for key, value in optional_args.items() if value is not None})

        try:
            return HuggingFaceEmbedding(**kwargs)
        except Exception as exc:
            guidance = (
                f"Failed to load local Hugging Face embedding model {self.model!r}. "
                "Confirm the model is available locally or that network access is available "
                "for the initial download. Rebuilding with a new embedding profile creates "
                "new Chroma collections; old collections remain until explicitly cleaned."
            )
            if self.model == QWEN_8B_MODEL:
                guidance += " Qwen3-Embedding-8B is heavy and may require substantial memory."
            raise RuntimeError(guidance) from exc


def build_embedding_backend(profile_name: str, config: dict[str, Any]) -> EmbeddingBackend:
    provider = str(config.get("provider", ""))
    if provider not in SUPPORTED_LOCAL_HUGGINGFACE_PROVIDERS:
        raise ValueError(
            f"Embedding provider {provider!r} is not implemented. "
            "Supported providers for this phase: local_huggingface. "
            "Voyage remains a future optional backend."
        )

    return LocalHuggingFaceEmbeddingBackend(name=profile_name, config=dict(config))


def embedding_profile_warnings(config: dict[str, Any]) -> list[str]:
    warnings = [
        "Changing embedding profiles creates model-specific Chroma collection names.",
        "Old Chroma collections remain until explicitly cleaned.",
    ]

    if config.get("model") == QWEN_8B_MODEL:
        warnings.append("Qwen3-Embedding-8B may be heavy and require substantial memory.")

    return warnings

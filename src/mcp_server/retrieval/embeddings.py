from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Protocol

QWEN_8B_MODEL = "Qwen/Qwen3-Embedding-8B"
SUPPORTED_LOCAL_HUGGINGFACE_PROVIDERS = {"local_huggingface"}
SUPPORTED_PROVIDERS = {"local_huggingface", "voyage"}


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


@dataclass(frozen=True)
class VoyageEmbeddingBackend:
    name: str
    config: dict[str, Any]

    @property
    def provider(self) -> str:
        return "voyage"

    @property
    def model(self) -> str:
        return str(self.config["model"])

    @property
    def dimension(self) -> int | str:
        return self.config.get("dimension", "default")

    def build_llamaindex_embedding(self) -> Any:
        api_key = os.environ.get("VOYAGE_API_KEY")
        if not api_key:
            raise RuntimeError(
                "VOYAGE_API_KEY is missing. Set it in .env before selecting a Voyage "
                "embedding profile."
            )

        try:
            from llama_index.core.embeddings import BaseEmbedding
        except Exception as exc:
            raise RuntimeError("LlamaIndex BaseEmbedding is unavailable.") from exc

        try:
            import voyageai  # type: ignore[import-not-found]
        except Exception as exc:
            raise RuntimeError(
                "The optional 'voyageai' package is required for Voyage embeddings. "
                "Install it before selecting a Voyage embedding profile."
            ) from exc

        model_name = self.model
        document_input_type = str(self.config.get("input_type_document", "document"))
        query_input_type = str(self.config.get("input_type_query", "query"))

        class VoyageLlamaIndexEmbedding(BaseEmbedding):  # type: ignore[misc]
            def _client(self) -> Any:
                return voyageai.Client(api_key=api_key)

            def _embed(self, texts: list[str], input_type: str) -> list[list[float]]:
                response = self._client().embed(
                    texts,
                    model=model_name,
                    input_type=input_type,
                )
                return [list(item) for item in response.embeddings]

            def _get_text_embedding(self, text: str) -> list[float]:
                return self._embed([text], document_input_type)[0]

            def _get_query_embedding(self, query: str) -> list[float]:
                return self._embed([query], query_input_type)[0]

            async def _aget_query_embedding(self, query: str) -> list[float]:
                return self._get_query_embedding(query)

            def _get_text_embeddings(self, texts: list[str]) -> list[list[float]]:
                return self._embed(texts, document_input_type)

        return VoyageLlamaIndexEmbedding(model_name=f"voyage:{model_name}")


def build_embedding_backend(profile_name: str, config: dict[str, Any]) -> EmbeddingBackend:
    provider = str(config.get("provider", ""))
    if provider == "local_huggingface":
        return LocalHuggingFaceEmbeddingBackend(name=profile_name, config=dict(config))
    if provider == "voyage":
        return VoyageEmbeddingBackend(name=profile_name, config=dict(config))
    if provider not in SUPPORTED_PROVIDERS:
        raise ValueError(
            f"Embedding provider {provider!r} is not implemented. "
            "Supported providers: local_huggingface, voyage."
        )

    raise ValueError(f"Unsupported embedding provider: {provider}")


def embedding_profile_warnings(config: dict[str, Any]) -> list[str]:
    warnings = [
        "Changing embedding profiles creates model-specific Chroma collection names.",
        "Old Chroma collections remain until explicitly cleaned.",
    ]

    if config.get("model") == QWEN_8B_MODEL:
        warnings.append("Qwen3-Embedding-8B may be heavy and require substantial memory.")
    if config.get("provider") == "voyage":
        warnings.append("Voyage uses a cloud API and requires strict token budget checks.")
        warnings.append("Voyage is optional and is not the default embedding backend.")

    return warnings

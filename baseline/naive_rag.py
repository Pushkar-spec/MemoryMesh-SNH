"""Naive RAG Baseline implementation (P8.1).

Governed by:
- docs/EVALUATION_AND_DEMO.md Section 1 (B0 Baseline) & Section 4
- docs/ROADMAP_AND_TASKS.md (T8.1)
- MEMORYMESH_MASTER_IMPLEMENTATION_PLAN_CANONICAL.md (P8.1)

Architecture:
- Embeds raw messages using FastEmbed (BAAI/bge-small-en-v1.5).
- Retrieves top 5 most relevant messages via cosine similarity.
- Bypasses MemoryMesh write-path resolution, temporal validity, and dependency graphs.
- Generates answers using the shared `baseline_answer` prompt and `FinalAnswer` schema.
- Routes all provider completions through `LLMClient.complete_json()`.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

import numpy as np

from memorymesh.errors import BaselineEmbeddingError
from memorymesh.llm.client import LLMClient, sanitize_sensitive_text
from memorymesh.llm.prompts import build_baseline_answer_prompt
from memorymesh.models import FinalAnswer, IngestMessage, format_datetime_utc

logger = logging.getLogger(__name__)


class NaiveRAGBaseline:
    """Naive RAG baseline system retrieving top-5 raw messages via FastEmbed."""

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        embedding_model: Any | None = None,
        *,
        config: Any | None = None,
        model: str | None = None,
        default_model: str = "claude-sonnet-5-5",
        retrieval_method: str = "fastembed",
    ) -> None:
        """Initialize naive RAG baseline.

        Args:
            llm_client: Configured LLMClient instance for completions.
            embedding_model: Optional pre-initialized or mock embedding model.
                             Defaults to FastEmbed TextEmbedding.
            config: Optional AppConfig instance to read default answer_model.
            model: Model name override for completions.
            default_model: Fallback answer model tier (matching MemoryMesh).
            retrieval_method: Method used for retrieval ("fastembed" or "lexical").
        """
        if retrieval_method not in ("fastembed", "lexical"):
            raise ValueError(f"Unsupported retrieval method: {retrieval_method}")

        self.retrieval_method = retrieval_method
        self.llm_client = llm_client
        if model is None and config is not None and hasattr(config, "engine"):
            model = config.engine.llm.answer_model
        self.model = model or default_model
        self._embedding_model = embedding_model
        self.messages: list[dict[str, Any]] = []
        self.embeddings: np.ndarray | None = None
        self._message_tokens: list[set[str]] = []

    @property
    def embedding_model(self) -> Any:
        """Lazy-load the FastEmbed TextEmbedding model if not injected."""
        if self._embedding_model is None:
            try:
                from fastembed import TextEmbedding

                self._embedding_model = TextEmbedding()
            except Exception as exc:
                logger.warning("Failed to initialize FastEmbed TextEmbedding: %s", exc)
                raise BaselineEmbeddingError(
                    f"FastEmbed embedding model is not available: {exc}. "
                    "Ensure fastembed is installed and weights are available, "
                    "or inject an embedding model."
                ) from exc
        return self._embedding_model

    def index_messages(self, messages: list[dict[str, Any] | IngestMessage]) -> None:
        """Index raw messages by generating and caching their normalized embeddings.

        Args:
            messages: List of raw message dictionaries or IngestMessage instances.
        """
        self.messages = []
        self._message_tokens = []
        if not messages:
            self.embeddings = None
            return

        normalized_msgs: list[dict[str, Any]] = []
        texts_to_embed: list[str] = []

        for idx, m in enumerate(messages):
            if isinstance(m, IngestMessage):
                m_dict = {
                    "id": f"m{idx + 1}",
                    "t": format_datetime_utc(m.event_time),
                    "source": m.source_id,
                    "text": m.text,
                }
            elif isinstance(m, dict):
                t_val = m.get("t") or m.get("event_time")
                if isinstance(t_val, datetime):
                    t_val = format_datetime_utc(t_val)
                m_dict = {
                    "id": str(m.get("id", f"m{idx + 1}")),
                    "t": str(t_val or ""),
                    "source": str(m.get("source") or m.get("source_id") or ""),
                    "text": str(m.get("text", "")),
                }
            else:
                m_dict = {
                    "id": f"m{idx + 1}",
                    "t": "",
                    "source": "",
                    "text": str(m),
                }

            normalized_msgs.append(m_dict)
            texts_to_embed.append(m_dict["text"])

        self.messages = normalized_msgs

        if self.retrieval_method == "fastembed":
            try:
                raw_embeddings = list(self.embedding_model.embed(texts_to_embed))
                emb_matrix = np.array(raw_embeddings, dtype=np.float32)

                # L2 normalize vectors for cosine similarity dot product
                norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
                norms[norms == 0.0] = 1.0
                self.embeddings = emb_matrix / norms
            except Exception as exc:
                logger.error("Failed to generate message embeddings: %s", exc)
                self.embeddings = None
                if isinstance(exc, BaselineEmbeddingError):
                    raise
                raise BaselineEmbeddingError(
                    f"FastEmbed embedding generation failed: {exc}"
                ) from exc

        elif self.retrieval_method == "lexical":
            # Explicit separate fallback mode (never reported as FastEmbed B0)
            self._message_tokens = [set(m["text"].lower().split()) for m in self.messages]

    def retrieve(self, query: str, top_k: int = 5) -> list[tuple[dict[str, Any], float]]:
        """Retrieve top_k messages most relevant to the query.

        Args:
            query: Question or search text.
            top_k: Maximum number of messages to return (default 5).

        Returns:
            List of (message_dict, similarity_score) sorted descending by similarity.
        """
        if not self.messages or len(self.messages) == 0:
            return []

        if self.retrieval_method == "fastembed":
            if self.embeddings is None:
                raise BaselineEmbeddingError(
                    "FastEmbed embeddings have not been computed. Call index_messages() first."
                )
            try:
                q_emb = list(self.embedding_model.embed([query]))[0]
                q_vec = np.array(q_emb, dtype=np.float32)
                q_norm = np.linalg.norm(q_vec)
                if q_norm > 0:
                    q_vec = q_vec / q_norm

                # Cosine similarities
                scores = np.dot(self.embeddings, q_vec)

                # Sort descending
                k = min(top_k, len(self.messages))
                top_indices = np.argsort(scores)[::-1][:k]

                return [(self.messages[idx], float(scores[idx])) for idx in top_indices]
            except Exception as exc:
                if isinstance(exc, BaselineEmbeddingError):
                    raise
                raise BaselineEmbeddingError(f"FastEmbed query embedding failed: {exc}") from exc

        elif self.retrieval_method == "lexical":
            # Explicit lexical fallback: token Jaccard similarity
            q_tokens = set(query.lower().split())
            scores = []
            for m_tokens in self._message_tokens:
                if not q_tokens or not m_tokens:
                    scores.append(0.0)
                else:
                    jaccard = len(q_tokens & m_tokens) / len(q_tokens | m_tokens)
                    scores.append(jaccard)
            scores_arr = np.array(scores, dtype=np.float32)
            k = min(top_k, len(self.messages))
            top_indices = np.argsort(scores_arr)[::-1][:k]
            return [(self.messages[idx], float(scores_arr[idx])) for idx in top_indices]

        raise ValueError(f"Unsupported retrieval method: {self.retrieval_method}")

    def ask(
        self,
        question: str,
        *,
        reference_time: datetime | str | None = None,
        top_k: int = 5,
        model: str | None = None,
    ) -> FinalAnswer:
        """Answer a question using retrieved top-5 messages and the baseline_answer prompt.

        Args:
            question: Natural language question.
            reference_time: Optional reference time string or datetime.
            top_k: Number of messages to retrieve (default 5).
            model: Optional model tier override.

        Returns:
            Validated FinalAnswer model.
        """
        # Handle empty collection
        if not self.messages:
            return FinalAnswer(
                value=None,
                values=[],
                conflict_flagged=False,
                needs_review_flagged=False,
                abstained=True,
                explanation="No messages indexed in naive RAG baseline.",
                answer_text="No messages available to answer the question.",
            )

        if self.llm_client is None:
            return FinalAnswer(
                value=None,
                values=[],
                conflict_flagged=False,
                needs_review_flagged=False,
                abstained=True,
                explanation="LLMClient is not configured for naive RAG baseline.",
                answer_text="LLMClient not configured.",
            )

        retrieved_items = self.retrieve(question, top_k=top_k)
        retrieved_msgs = [item[0] for item in retrieved_items]

        # Format prompt
        system_prompt, user_prompt = build_baseline_answer_prompt(
            question=question,
            messages=retrieved_msgs,
            reference_time=reference_time,
        )

        eff_model = model or self.model

        try:
            proposal = self.llm_client.complete_json(
                prompt_id="baseline_answer",
                system=system_prompt,
                user=user_prompt,
                response_model=FinalAnswer,
                model=eff_model,
            )
            return proposal
        except Exception as exc:
            sanitized_err = sanitize_sensitive_text(str(exc), max_len=200)
            logger.warning("Baseline answer generation failed: %s", sanitized_err)
            return FinalAnswer(
                value=None,
                values=[],
                conflict_flagged=False,
                needs_review_flagged=False,
                abstained=True,
                explanation=f"Baseline answer generation failed: {sanitized_err}",
                answer_text="Failed to generate baseline answer.",
            )

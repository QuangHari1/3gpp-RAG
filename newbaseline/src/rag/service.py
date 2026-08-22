"""One-round, paper-compatible Telco-oRAG orchestration."""

from __future__ import annotations

import numpy as np

from ..settings import Settings, load_settings
from ..embeddings import create_embedding_provider
from .abbreviation_resolver import ContextualAbbreviationResolver
from .clients import OpenAICompatibleRagClient, RagClient
from .corpus import PaperEmbeddingCorpus
from .router import PaperNNRouter, SemanticSeriesRouter
from .types import RagResult
from .vocabulary import Vocabulary


class PaperRagService:
    """Question -> rephrase -> vocabulary -> router -> FAISS -> optional answer."""

    def __init__(
        self,
        settings: Settings | None = None,
        client: RagClient | None = None,
        router: PaperNNRouter | None = None,
        corpus: PaperEmbeddingCorpus | None = None,
        vocabulary: Vocabulary | None = None,
    ) -> None:
        self.settings = settings or load_settings()
        resources = self.settings.workspace_root / self.settings.get("rag", "resources_dir")
        embedding_root = (
            self.settings.dataset_dir
            / "3gpp"
            / "Embeddings"
            / f"Rel-{self.settings.release}"
            / self.settings.get("rag", "selection_id")
        )
        embedding_provider = create_embedding_provider(self.settings) if client is None else None
        self.client = client or OpenAICompatibleRagClient(
            embedding_provider=embedding_provider,
            rephrase_model=self.settings.get("rag", "rephrase_model"),
            answer_model=self.settings.get("rag", "answer_model"),
            llm_provider=self.settings.get("llm", "provider"),
            api_key_env=self.settings.get("llm", "api_key_env"),
            base_url=self.settings.get("llm", "base_url"),
            thinking_mode=self.settings.get("llm", "thinking_mode"),
            temperature=self.settings.get("llm", "temperature"),
        )
        if router is not None:
            self.router = router
        elif self.settings.get("rag", "router_backend") == "paper_nn":
            if (
                self.settings.get("embedding", "backend") != "openai"
                or self.settings.get("embedding", "model") != "text-embedding-3-large"
                or self.settings.get("embedding", "dimensions") != 1024
            ):
                raise RuntimeError(
                    "rag.router_backend='paper_nn' requires OpenAI text-embedding-3-large at 1024 dimensions. "
                    "Use router_backend='semantic' for another embedding model."
                )
            self.router = PaperNNRouter(
                resources / self.settings.get("rag", "router_checkpoint"),
                resources / self.settings.get("rag", "series_descriptions"),
                self.settings.get("rag", "router_similarity_scale"),
            )
        elif self.settings.get("rag", "router_backend") == "semantic":
            if embedding_provider is None:
                raise ValueError("Inject a router when constructing PaperRagService with a custom client.")
            self.router = SemanticSeriesRouter(
                resources / self.settings.get("rag", "series_descriptions"), embedding_provider.embed
            )
        else:
            raise ValueError(f"Unsupported [rag].router_backend: {self.settings.get('rag', 'router_backend')}")
        self.corpus = corpus or PaperEmbeddingCorpus(embedding_root, self.settings.workspace_root)
        vocabulary_mode = self.settings.get("vocabulary", "mode")
        self._contextual_abbreviation_resolution = vocabulary_mode == "release18_contextual"
        self.abbreviation_resolver = ContextualAbbreviationResolver()
        if vocabulary is not None:
            self.vocabulary = vocabulary
        elif vocabulary_mode == "paper_legacy":
            self.vocabulary = Vocabulary.from_docx(resources / self.settings.get("rag", "vocabulary"))
        elif vocabulary_mode in {"release18_unambiguous", "release18_contextual"}:
            self.vocabulary = Vocabulary.from_release18_assets(
                resources / self.settings.get("vocabulary", "definitions_file"),
                resources / self.settings.get("vocabulary", "abbreviations_file"),
            )
        else:
            raise ValueError(f"Unsupported [vocabulary].mode: {vocabulary_mode}")

    def run(
        self,
        question: str,
        include_answer: bool = True,
        answer_prompt: str | None = None,
        strict_multiple_choice: bool = False,
    ) -> RagResult:
        """Retrieve for ``question`` and optionally answer a richer prompt.

        TeleQnA uses the plain question for routing/retrieval while supplying
        its answer choices only to the final answer model.
        """
        if not question.strip():
            raise ValueError("Question must not be empty.")
        rephrased = self.client.rephrase(question)
        enriched = self.vocabulary.enrich(rephrased)
        query_embedding = np.asarray(self.client.embed(enriched), dtype=np.float32)
        selected = self.router.route(query_embedding, self.settings.get("rag", "router_top_k"))
        seed_retrievals, searched, empty = self.corpus.search(
            selected, query_embedding, self.settings.get("rag", "retrieval_top_k")
        )
        abbreviation_resolutions = []
        if getattr(self, "_contextual_abbreviation_resolution", False):
            matches = self.vocabulary.ambiguous_matches(
                rephrased,
                self.settings.get("vocabulary", "contextual_candidate_limit"),
                self.settings.get("vocabulary", "contextual_excluded_acronyms"),
            )
            if matches:
                candidate_queries = self.abbreviation_resolver.candidate_queries(rephrased, matches)
                candidate_embeddings = np.asarray(
                    self.client.embed_many([candidate.render(rephrased) for candidate in candidate_queries]),
                    dtype=np.float32,
                )
                candidate_hits_by_query, _, _ = self.corpus.search_many(
                    selected,
                    candidate_embeddings,
                    self.settings.get("vocabulary", "contextual_support_top_k"),
                )
                candidate_hits = {
                    (candidate.acronym, candidate.candidate.expansion): hits
                    for candidate, hits in zip(candidate_queries, candidate_hits_by_query, strict=True)
                }
                abbreviation_resolutions = self.abbreviation_resolver.resolve(
                    matches,
                    candidate_hits,
                    seed_retrievals,
                    selected,
                    self.settings.get("vocabulary", "contextual_min_score"),
                    self.settings.get("vocabulary", "contextual_min_margin"),
                )
                resolved_abbreviations = {
                    resolution.acronym: resolution.selected_expansion
                    for resolution in abbreviation_resolutions
                    if resolution.selected_expansion is not None
                }
                if resolved_abbreviations:
                    enriched = self.vocabulary.enrich(rephrased, resolved_abbreviations)
                    query_embedding = np.asarray(self.client.embed(enriched), dtype=np.float32)
                    selected = self.router.route(query_embedding, self.settings.get("rag", "router_top_k"))
                    seed_retrievals, searched, empty = self.corpus.search(
                        selected, query_embedding, self.settings.get("rag", "retrieval_top_k")
                    )
        retrievals, citation_paths = self.corpus.expand_citations(
            seed_retrievals,
            max_depth=self.settings.get("rag", "citation_max_depth"),
            total_limit=self.settings.get("rag", "citation_total_chunks"),
            chunks_per_heading=self.settings.get("rag", "citation_chunks_per_heading"),
            query_embedding=query_embedding,
            embed_many=self.client.embed_many,
        )
        answer = None
        if include_answer:
            contexts = [self._format_context(hit) for hit in retrievals]
            answer = self.client.answer(
                answer_prompt or question,
                contexts,
                strict_multiple_choice=strict_multiple_choice,
            )
        return RagResult(
            question=question,
            rephrased_query=rephrased,
            enriched_query=enriched,
            router_selected_series=selected,
            empty_selected_series=empty,
            searched_series=searched,
            retrievals=retrievals,
            answer=answer,
            citation_paths=citation_paths,
            abbreviation_resolutions=abbreviation_resolutions,
        )

    @staticmethod
    def _format_context(hit) -> str:
        """Keep LLM provenance compact; full citation metadata remains in the trace."""
        metadata = hit.metadata
        source = (
            f"series={hit.series}; document={metadata.get('document_id')}; "
            f"heading={metadata.get('heading')}; chunk_id={metadata.get('chunk_id')}"
        )
        if hit.origin == "citation":
            source += f"; citation_depth={hit.citation_depth}; parent={hit.parent_chunk_id}"
        return f"[{source}]\n{hit.text}"

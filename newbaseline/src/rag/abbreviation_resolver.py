"""Evidence-based disambiguation of Release-18 abbreviation candidates."""

from __future__ import annotations

from dataclasses import dataclass

from .types import AbbreviationResolution, RetrievalHit
from .vocabulary import AmbiguousAbbreviationCandidate


@dataclass(frozen=True)
class CandidateQuery:
    acronym: str
    candidate: AmbiguousAbbreviationCandidate

    def render(self, question: str) -> str:
        return f"{question}\n\nCandidate abbreviation meaning:\n{self.acronym}: {self.candidate.expansion}"


class ContextualAbbreviationResolver:
    """Select a sense only when semantic evidence separates it from alternatives."""

    def candidate_queries(
        self, question: str, matches: dict[str, list[AmbiguousAbbreviationCandidate]]
    ) -> list[CandidateQuery]:
        return [
            CandidateQuery(acronym, candidate)
            for acronym, candidates in matches.items()
            for candidate in candidates
        ]

    def resolve(
        self,
        matches: dict[str, list[AmbiguousAbbreviationCandidate]],
        candidate_hits: dict[tuple[str, str], list[RetrievalHit]],
        seed_hits: list[RetrievalHit],
        selected_series: list[str],
        min_score: float,
        min_margin: float,
    ) -> list[AbbreviationResolution]:
        if not 0 <= min_score <= 1 or min_margin < 0:
            raise ValueError("Invalid abbreviation-resolution confidence thresholds")
        seed_text = "\n".join(hit.text.casefold() for hit in seed_hits)
        selected = set(selected_series)
        resolutions: list[AbbreviationResolution] = []
        for acronym, candidates in matches.items():
            scored: list[dict[str, object]] = []
            for candidate in candidates:
                hits = candidate_hits.get((acronym, candidate.expansion), [])
                semantic_score = hits[0].score if hits else 0.0
                evidence_score = 1.0 if candidate.expansion.casefold() in seed_text else 0.0
                series_prior = (
                    len(selected.intersection(candidate.source_series)) / len(candidate.source_series)
                    if candidate.source_series
                    else 0.0
                )
                score = 0.70 * semantic_score + 0.20 * evidence_score + 0.10 * series_prior
                scored.append(
                    {
                        "expansion": candidate.expansion,
                        "semantic_score": semantic_score,
                        "evidence_score": evidence_score,
                        "series_prior": series_prior,
                        "score": score,
                        "source_series": list(candidate.source_series),
                    }
                )
            scored.sort(key=lambda row: (-float(row["score"]), str(row["expansion"]).casefold()))
            winner = scored[0] if scored else None
            runner_up = scored[1] if len(scored) > 1 else None
            margin = float(winner["score"]) - float(runner_up["score"]) if runner_up else float("inf")
            selected_expansion = (
                str(winner["expansion"])
                if winner and float(winner["score"]) >= min_score and margin >= min_margin
                else None
            )
            resolutions.append(
                AbbreviationResolution(
                    acronym=acronym,
                    candidates=scored,
                    selected_expansion=selected_expansion,
                    confidence=float(winner["score"]) if winner else 0.0,
                    margin=margin if margin != float("inf") else None,
                )
            )
        return resolutions

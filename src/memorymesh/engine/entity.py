"""Pure deterministic entity resolution logic for MemoryMesh.

Governed by docs/ARCHITECTURE.md Section 3 (Stage 2), docs/ROADMAP_AND_TASKS.md (T2.4),
and config/entities.yaml.

Functional Core:
  - Pure functions only.
  - No database operations, no repository imports.
  - No datetime.now() calls.
  - Deterministic two-tier matching:
      Tier 1: Exact normalized alias matching (case-insensitive, whitespace-normalized).
      Tier 2: Fuzzy matching via rapidfuzz token_set_ratio.
  - Ambiguous top-score ties are never merged automatically; they create review-flagged entities.
"""

from __future__ import annotations

from enum import StrEnum
from typing import NamedTuple

from rapidfuzz import fuzz, utils


def normalize_entity_name(name: str) -> str:
    """Normalize an entity name by trimming and collapsing internal whitespace to lowercase."""
    return " ".join(name.strip().split()).lower()


class EntityResolutionAction(StrEnum):
    EXACT_MATCH = "exact_match"
    FUZZY_MATCH = "fuzzy_match"
    AMBIGUOUS_MATCH = "ambiguous_match"
    REVIEW_MATCH = "review_match"
    UNKNOWN = "unknown"


class EntityCandidate(NamedTuple):
    canonical_name: str
    entity_type: str
    aliases: list[str]
    db_id: int | None = None


class EntityResolutionResult(NamedTuple):
    action: EntityResolutionAction
    canonical_name: str
    entity_type: str
    matched_entity_id: int | None
    needs_review: bool
    score: float
    matched_candidate_name: str | None = None


def resolve_entity(
    subject: str,
    candidates: list[EntityCandidate],
    auto_match_threshold: float = 90.0,
    tiebreak_min_threshold: float = 75.0,
) -> EntityResolutionResult:
    """Pure entity resolution matching subject against provided candidates.

    Tier 1 (Exact):
      - Checks normalized subject against candidate canonical names and aliases.
      - Returns exact match deterministically if found.

    Tier 2 (Fuzzy):
      - Scores subject against all candidates using rapidfuzz token_set_ratio.
      - If multiple distinct candidates tie for the top score (score >= tiebreak_min_threshold),
        returns AMBIGUOUS_MATCH with needs_review=True (does not auto-merge).
      - If a single candidate achieves score >= auto_match_threshold:
        returns FUZZY_MATCH with needs_review=False (auto-merges to that candidate).
      - If a single candidate achieves tiebreak_min_threshold <= score < auto_match_threshold:
        returns REVIEW_MATCH with needs_review=True (creates separate flagged entity).
      - If score < tiebreak_min_threshold:
        returns UNKNOWN with needs_review=False (creates new unflagged entity).
    """
    norm_subject = normalize_entity_name(subject)
    if not norm_subject:
        raise ValueError("Cannot resolve empty entity name.")

    # -------------------------------------------------------------------------
    # Tier 1: Exact normalized match against candidates
    # -------------------------------------------------------------------------
    exact_matches: list[EntityCandidate] = []
    seen_candidates: set[tuple[str, int | None]] = set()

    for cand in candidates:
        cand_key = (cand.canonical_name, cand.db_id)
        if cand_key in seen_candidates:
            continue

        matched = False
        if norm_subject == normalize_entity_name(cand.canonical_name):
            matched = True
        else:
            for alias in cand.aliases:
                if norm_subject == normalize_entity_name(alias):
                    matched = True
                    break

        if matched:
            exact_matches.append(cand)
            seen_candidates.add(cand_key)

    if len(exact_matches) > 1:
        # Multiple distinct candidates match the same normalized name/alias:
        # Exact collision is ambiguous. Do not choose by candidate ordering or auto-merge.
        return EntityResolutionResult(
            action=EntityResolutionAction.AMBIGUOUS_MATCH,
            canonical_name=subject,
            entity_type="unknown",
            matched_entity_id=None,
            needs_review=True,
            score=100.0,
            matched_candidate_name=None,
        )

    if len(exact_matches) == 1:
        single_exact = exact_matches[0]
        return EntityResolutionResult(
            action=EntityResolutionAction.EXACT_MATCH,
            canonical_name=single_exact.canonical_name,
            entity_type=single_exact.entity_type,
            matched_entity_id=single_exact.db_id,
            needs_review=False,
            score=100.0,
            matched_candidate_name=single_exact.canonical_name,
        )

    # -------------------------------------------------------------------------
    # Tier 2: Fuzzy matching against candidates
    # -------------------------------------------------------------------------
    if not candidates:
        return EntityResolutionResult(
            action=EntityResolutionAction.UNKNOWN,
            canonical_name=subject,
            entity_type="unknown",
            matched_entity_id=None,
            needs_review=False,
            score=0.0,
            matched_candidate_name=None,
        )

    # Calculate best score per candidate
    candidate_scores: list[tuple[float, EntityCandidate]] = []
    for cand in candidates:
        best_cand_score = 0.0
        all_names = [cand.canonical_name, *cand.aliases]
        for name in all_names:
            score = float(
                fuzz.token_set_ratio(
                    norm_subject,
                    name,
                    processor=utils.default_process,
                )
            )
            if score > best_cand_score:
                best_cand_score = score
        candidate_scores.append((best_cand_score, cand))

    max_score = max(cs[0] for cs in candidate_scores)

    # Find all candidates with the top score
    top_candidates = [cand for score, cand in candidate_scores if score == max_score]

    # Ambiguity check: if multiple distinct candidates tie for the top score
    # and the score is at or above the review threshold (tiebreak_min_threshold):
    # Do NOT merge automatically! Create a separate entity flagged for review.
    if len(top_candidates) > 1 and max_score >= tiebreak_min_threshold:
        return EntityResolutionResult(
            action=EntityResolutionAction.AMBIGUOUS_MATCH,
            canonical_name=subject,
            entity_type="unknown",
            matched_entity_id=None,
            needs_review=True,
            score=max_score,
            matched_candidate_name=None,
        )

    # Single top candidate
    single_cand = top_candidates[0]

    if max_score >= auto_match_threshold:
        return EntityResolutionResult(
            action=EntityResolutionAction.FUZZY_MATCH,
            canonical_name=single_cand.canonical_name,
            entity_type=single_cand.entity_type,
            matched_entity_id=single_cand.db_id,
            needs_review=False,
            score=max_score,
            matched_candidate_name=single_cand.canonical_name,
        )

    if max_score >= tiebreak_min_threshold:
        return EntityResolutionResult(
            action=EntityResolutionAction.REVIEW_MATCH,
            canonical_name=subject,
            entity_type="unknown",
            matched_entity_id=None,
            needs_review=True,
            score=max_score,
            matched_candidate_name=None,
        )

    # Below review threshold: new unknown entity, unflagged
    return EntityResolutionResult(
        action=EntityResolutionAction.UNKNOWN,
        canonical_name=subject,
        entity_type="unknown",
        matched_entity_id=None,
        needs_review=False,
        score=max_score,
        matched_candidate_name=None,
    )

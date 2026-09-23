"""Candidate search — keep several formula versions alive and choose between them on held-out data.

The trainer's loop commits every proposal: `FormulaOptimizer.step()` mutates the formula in place and
returns nothing, so nothing outside it can decline an edit, compare two edits, or keep the original
around. This package adds that outer layer. A `Candidate` is an immutable snapshot with lineage, an
`EvaluationStore` keeps results per item rather than one scalar per epoch, and a `SearchController`
evaluates a parent and its child on the same items, gates on that paired comparison, and scores only
admitted candidates on a split the proposer never saw.

Parent selection, item sampling, the gate and stopping are plain callables, so a different search
strategy is a different function rather than a subclass.
"""

from .candidate import Candidate
from .controller import (
    CandidateGenerator,
    EvaluationPass,
    IterationRecord,
    SearchController,
    SearchResult,
)
from .evaluation import EvaluationRecord, EvaluationStore, RolloutStatus
from .guards import GuardVerdict, all_of, forbid_patterns, forbid_values
from .policies import (
    bucket_items,
    budget_and_saturation,
    force_sequence,
    pareto_frontier,
    replay_sampler,
    select_by_mean,
    select_from_pareto_frontier,
    stratified_sampler,
    strict_improvement,
)
from .views import FeedbackView, minimal_view

__all__ = [
    "Candidate",
    "CandidateGenerator",
    "EvaluationPass",
    "EvaluationRecord",
    "EvaluationStore",
    "FeedbackView",
    "GuardVerdict",
    "IterationRecord",
    "RolloutStatus",
    "SearchController",
    "SearchResult",
    "all_of",
    "bucket_items",
    "budget_and_saturation",
    "force_sequence",
    "forbid_patterns",
    "forbid_values",
    "minimal_view",
    "pareto_frontier",
    "replay_sampler",
    "select_by_mean",
    "select_from_pareto_frontier",
    "stratified_sampler",
    "strict_improvement",
]

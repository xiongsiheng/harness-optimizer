"""SearchController: the state machine the upstream Trainer does not have.

Upstream's loop accumulates rollouts, calls `optimizer.step()` once per epoch, and commits
whatever comes back. This controller keeps a population, gates each proposal against its parent
on the same items, and evaluates admitted candidates on a split the proposer never saw. It
coordinates only: rollout execution, scoring, and formula editing all stay where they are.

One iteration, in order:

    parent  = parent_selection(pool, store)
    batch   = item_sampling(iteration, parent, store, feedback_items)
    evaluate(parent, batch, role="feedback")
    view    = feedback_view(rollouts, records)
    child   = generator.propose(parent, view)
    guard(child)                                    # blocked proposals cost no rollouts
    evaluate(child, batch, role="feedback")          # the same items — a paired comparison
    gate(parent, child, store, batch)
        rejected -> record and continue, at no selection cost
        admitted -> evaluate(child, selection_items, role="selection"); add to the pool
    stopping(...)

The result is whichever pool member the selection policy prefers, which may be the seed.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, NamedTuple, Protocol, Sequence

from ..datamodels import Reward, Rollout
from ..formulas import Formula
from ..rewards import RewardFunction
from ..rollout_engines import AgentRolloutEngine
from .candidate import Candidate
from .evaluation import EvaluationRecord, EvaluationStore, RolloutStatus
from .policies import FEEDBACK_ROLE, SELECTION_ROLE


class CandidateGenerator(Protocol):
    """Proposes a new parameter dict from a parent and a view of its trajectories.

    Deliberately *not* the upstream `FormulaOptimizer` interface: that one mutates the formula
    and returns nothing, which is precisely what makes a proposal impossible to decline. An
    existing optimizer is adapted by applying its in-place update to a copy of the parent's
    params, which needs no upstream change.
    """

    def propose(self, parent: Candidate, view: Any) -> Mapping[str, Any]: ...


class EvaluationPass(NamedTuple):
    """One evaluation of one candidate on one role: the records stored, and the rollouts they came
    from. Both are needed — the store answers "how did it score", the rollouts are what the
    generator reads."""

    records: list["EvaluationRecord"]
    rollouts: list


@dataclass
class IterationRecord:
    iteration: int
    parent_id: str
    child_id: str | None
    batch_size: int
    gate: dict
    admitted: bool
    selection_mean: float | None
    rollouts_used: int
    wall_seconds: float
    notes: dict = field(default_factory=dict)


@dataclass
class SearchResult:
    best: Candidate
    pool: list[Candidate]
    iterations: list[IterationRecord]
    rollouts_used: int
    stop_reason: str

    def to_json(self) -> dict:
        return {
            "best": self.best.candidate_id,
            "pool": [
                {"id": c.candidate_id, "parent": c.parent_id, "origin": c.origin, "size": c.size()}
                for c in self.pool
            ],
            "iterations": [vars(i) for i in self.iterations],
            "rollouts_used": self.rollouts_used,
            "stop_reason": self.stop_reason,
        }


class SearchController:
    """Owns the population, the evaluation store, the budget and the stopping state."""

    def __init__(
        self,
        seed: Candidate,
        generator: CandidateGenerator,
        engine: AgentRolloutEngine,
        reward_fn: RewardFunction,
        *,
        feedback_items: Sequence[str],
        selection_items: Sequence[str],
        parent_selection: Callable[[Sequence[Candidate], EvaluationStore], Candidate],
        final_selection: Callable[[Sequence[Candidate], EvaluationStore], Candidate] | None = None,
        item_sampling: Callable[..., list[str]],
        gate: Callable[..., dict],
        stopping: Callable[..., dict],
        store: EvaluationStore | None = None,
        formula: Formula | None = None,
        feedback_view: Callable[[Sequence[Rollout], Sequence[EvaluationRecord]], Any] | None = None,
        status_from: Callable[[Rollout, Reward], RolloutStatus] | None = None,
        proposal_guard: Callable[[Mapping[str, Any]], Any] | None = None,
        evaluate_seed_on_selection: bool = True,
    ):
        self.pool: list[Candidate] = [seed]
        self.generator = generator
        self.engine = engine
        self.reward_fn = reward_fn
        self.feedback_items = list(feedback_items)
        self.selection_items = list(selection_items)
        self.parent_selection = parent_selection
        # Which candidate to extend and which candidate to return are separate decisions. They
        # coincide by default, but a caller replaying a recorded run forces the parent while still
        # wanting the real selector to pick the answer — and a caller could reasonably extend
        # greedily while returning the most robust member.
        self.final_selection = final_selection or parent_selection
        self.item_sampling = item_sampling
        self.gate = gate
        self.stopping = stopping
        self.store = store if store is not None else EvaluationStore()
        self.formula = formula
        self.feedback_view = feedback_view or (lambda rollouts, records: list(rollouts))
        self.status_from = status_from or _default_status
        # Checked before the child is evaluated, so a rejected proposal costs nothing. The guard
        # sees the proposal; the generator does not see what the guard knows.
        self.proposal_guard = proposal_guard
        self.evaluate_seed_on_selection = evaluate_seed_on_selection
        self.rollouts_used = 0
        self.iterations: list[IterationRecord] = []
        self._next_index = 1

    # ------------------------------------------------------------------ internals
    def _materialise(self, candidate: Candidate) -> None:
        """Point the (mutable) Formula at this candidate before any rollout is requested.

        Upstream binds the formula to the engine at construction, so with several candidates
        alive there is exactly one live parameter set. Materialising immediately before a request
        and asserting afterwards is what keeps "which candidate is executing" unambiguous.
        """
        if self.formula is None:
            return
        self.formula.update_params(dict(candidate.params))
        now = self.formula.get_tunable_params()
        for k, v in candidate.params.items():
            if now.get(k) != v:
                raise RuntimeError(
                    f"formula did not take parameter {k!r} for candidate {candidate.candidate_id}; "
                    "the engine would have executed a different candidate than the one requested"
                )
        self.engine.ensure_sync_params()

    def _evaluate(self, candidate: Candidate, items: Sequence[str], role: str) -> EvaluationPass:
        self._materialise(candidate)
        # The candidate's parameters travel with the request. An engine that has to look them up
        # elsewhere can get them wrong: reaching into the pool fails for a child, because a child is
        # not in the pool until it is admitted, and the engine then silently executes the seed. That
        # happened, and it produced a whole iteration in which parent and child ran the same empty
        # document.
        params = dict(candidate.params)
        samples = [
            {"item_id": i, "candidate_id": candidate.candidate_id, "role": role, "params": params}
            for i in items
        ]
        rollouts = list(self.engine.generate_batch(samples))
        if not rollouts:
            raise RuntimeError(
                f"no rollouts returned for {candidate.candidate_id} on role {role!r} "
                f"({len(items)} items requested) — an empty evaluation must fail loudly rather "
                "than be read as a saturated search"
            )
        seen: dict[str, int] = {}
        records = []
        for ro in rollouts:
            rw = self.reward_fn(ro)
            item = str(ro.data_sample.get("item_id"))
            rep = seen.get(item, 0)
            seen[item] = rep + 1
            records.append(
                EvaluationRecord(
                    candidate_id=candidate.candidate_id,
                    role=role,
                    item_id=item,
                    replicate=rep,
                    score=float(rw.reward),
                    status=self.status_from(ro, rw),
                    metrics={**dict(ro.metrics or {}), **dict(rw.metadata or {})},
                )
            )
        self.store.extend(records)
        self.rollouts_used += len(rollouts)
        # The rollouts travel back with the records. Returning records alone loses everything the
        # generator reflects on — transcripts, predictions, call counts — and the loss is silent:
        # the proposer still writes a document from the remaining prompt sections, so nothing errors
        # and the run reports a healthy accept. Three P3 runs were spent that way.
        return EvaluationPass(records=records, rollouts=rollouts)

    # ----------------------------------------------------------------------- driving
    def run(self, max_iterations: int | None = None) -> SearchResult:
        if self.evaluate_seed_on_selection and not self.store.has(
            self.pool[0].candidate_id, SELECTION_ROLE
        ):
            self._evaluate(self.pool[0], self.selection_items, SELECTION_ROLE)

        stop_reason = "completed"
        iteration = 0
        while True:
            iteration += 1
            if max_iterations is not None and iteration > max_iterations:
                stop_reason = f"iteration cap {max_iterations}"
                break
            parent = self.parent_selection(self.pool, self.store)
            verdict = self.stopping(
                iteration=iteration,
                parent=parent,
                store=self.store,
                rollouts_used=self.rollouts_used,
                items=self.feedback_items,
            )
            if verdict["stop"]:
                stop_reason = verdict["reason"]
                break

            t0 = time.time()
            # Feedback is scoped to the iteration. A candidate can appear as a child in one
            # iteration and as the parent in the next, and pooling both batches under one role
            # would average over different item sets — which changes the gate's comparison.
            fb_role = f"{FEEDBACK_ROLE}:{iteration}"
            batch = self.item_sampling(iteration, parent, self.store, self.feedback_items)
            p = self._evaluate(parent, batch, fb_role)
            view = self.feedback_view(p.rollouts, p.records)
            child_params = self.generator.propose(parent, view)
            child = parent.child(f"c{self._next_index}", child_params, iteration=iteration)

            if self.proposal_guard is not None:
                verdict = self.proposal_guard(dict(child_params))
                if not verdict.ok:
                    self.iterations.append(
                        IterationRecord(
                            iteration=iteration,
                            parent_id=parent.candidate_id,
                            child_id=None,
                            batch_size=len(batch),
                            gate={
                                "admit": False,
                                "blocked_by_guard": True,
                                "guard": verdict.guard,
                                "violations": verdict.violations[:20],
                            },
                            admitted=False,
                            selection_mean=None,
                            rollouts_used=self.rollouts_used,
                            wall_seconds=round(time.time() - t0, 3),
                            notes={
                                "guard_summary": verdict.summary(),
                                "rejected_child_size": child.size(),
                            },
                        )
                    )
                    continue

            self._evaluate(child, batch, fb_role)
            g = self.gate(parent, child, self.store, batch, role_override=fb_role)

            sel_mean = None
            if g["admit"]:
                self._next_index += 1
                self.pool.append(child)
                self._evaluate(child, self.selection_items, SELECTION_ROLE)
                sel_mean = self.store.mean(child.candidate_id, SELECTION_ROLE)

            self.iterations.append(
                IterationRecord(
                    iteration=iteration,
                    parent_id=parent.candidate_id,
                    child_id=child.candidate_id if g["admit"] else None,
                    batch_size=len(batch),
                    gate=g,
                    admitted=bool(g["admit"]),
                    selection_mean=sel_mean,
                    rollouts_used=self.rollouts_used,
                    wall_seconds=round(time.time() - t0, 3),
                    notes={"rejected_child_size": None if g["admit"] else child.size()},
                )
            )

        best = self.final_selection(self.pool, self.store)
        return SearchResult(
            best=best,
            pool=list(self.pool),
            iterations=list(self.iterations),
            rollouts_used=self.rollouts_used,
            stop_reason=stop_reason,
        )


def _default_status(rollout: Rollout, reward: Reward) -> RolloutStatus:
    """Read a status the rollout engine already put in metadata, else assume valid.

    Engines know why a run failed; a reward function usually does not. So the engine is expected
    to report it, and this default only keeps the controller usable when it does not.
    """
    for src in (reward.metadata or {}, rollout.metrics or {}, rollout.metadata or {}):
        s = src.get("status")
        if s:
            return s if isinstance(s, RolloutStatus) else RolloutStatus(str(s))
    return RolloutStatus.VALID

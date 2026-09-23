"""The controller's contract, on synthetic score tables.

Each test states a situation the upstream greedy loop gets wrong, and asserts the search loop gets
it right: a proposal can be declined, the seed can win, a rejected proposal costs no selection
rollouts, and an empty evaluation is an error rather than a saturated search.
"""

from __future__ import annotations

import pytest

from strands_harness_optimizer.search.candidate import Candidate
from strands_harness_optimizer.search.controller import SearchController
from strands_harness_optimizer.search.evaluation import EvaluationStore
from strands_harness_optimizer.search.policies import (
    budget_and_saturation,
    replay_sampler,
    select_by_mean,
    select_from_pareto_frontier,
    strict_improvement,
)
from strands_harness_optimizer.search.views import minimal_view
from tests.search.helpers import DictFormula, ScoreReward, ScriptedGenerator, TableEngine

ITEMS = ["i1", "i2"]
SEL = ["s1", "s2"]


def build(
    engine, generator, *, iterations=1, parent=None, formula=None, view=None, gate=None, store=None
):
    return SearchController(
        seed=Candidate.seed({"document": ""}),
        generator=generator,
        engine=engine,
        reward_fn=ScoreReward(),
        feedback_items=ITEMS,
        selection_items=SEL,
        parent_selection=parent or select_by_mean(),
        final_selection=select_by_mean(),
        item_sampling=replay_sampler({i + 1: ITEMS for i in range(iterations)}),
        gate=gate or strict_improvement(),
        stopping=budget_and_saturation(max_iterations=iterations),
        store=store or EvaluationStore(),
        formula=formula,
        feedback_view=view,
    )


def test_a_worse_proposal_is_declined_and_the_seed_is_returned():
    """The situation the upstream loop cannot express: the seed was already the best answer."""
    engine = TableEngine(
        table={
            ("c0", "i1"): [1.0],
            ("c0", "i2"): [1.0],
            ("c1", "i1"): [0.0],
            ("c1", "i2"): [0.0],
            ("c0", "s1"): [1.0],
            ("c0", "s2"): [1.0],
        },
        default={"c1": [0.0]},
    )
    ctl = build(engine, ScriptedGenerator([{"document": "worse"}]))
    res = ctl.run()
    assert [it.admitted for it in res.iterations] == [False]
    assert res.best.candidate_id == "c0"
    assert len(res.pool) == 1, "a declined proposal must not enter the population"


def test_a_rejected_proposal_costs_no_selection_rollouts():
    engine = TableEngine(
        table={("c0", "i1"): [1.0], ("c0", "i2"): [1.0]}, default={"c0": [1.0], "c1": [0.0]}
    )
    ctl = build(engine, ScriptedGenerator([{"document": "worse"}]))
    ctl.run()
    selection_calls = [c for c in engine.calls if c[1] == "selection"]
    assert [c[0] for c in selection_calls] == [
        "c0"
    ], "only the seed's selection sweep should have been paid for"


def test_an_improvement_is_admitted_and_evaluated_on_the_selection_set():
    engine = TableEngine(
        table={
            ("c0", "i1"): [0.0],
            ("c0", "i2"): [0.0],
            ("c1", "i1"): [1.0],
            ("c1", "i2"): [1.0],
            ("c0", "s1"): [0.0],
            ("c0", "s2"): [0.0],
            ("c1", "s1"): [1.0],
            ("c1", "s2"): [1.0],
        }
    )
    ctl = build(engine, ScriptedGenerator([{"document": "better"}]))
    res = ctl.run()
    assert [it.admitted for it in res.iterations] == [True]
    assert res.best.candidate_id == "c1"
    assert ctl.store.has("c1", "selection")
    assert res.iterations[0].selection_mean == pytest.approx(1.0)


def test_feedback_gain_that_does_not_transfer_still_admits_but_does_not_win():
    """The measured failure mode: better on the batch it was shown, worse on held-out items."""
    engine = TableEngine(
        table={
            ("c0", "i1"): [0.0],
            ("c0", "i2"): [0.0],
            ("c1", "i1"): [1.0],
            ("c1", "i2"): [1.0],
            ("c0", "s1"): [1.0],
            ("c0", "s2"): [1.0],
            ("c1", "s1"): [0.0],
            ("c1", "s2"): [0.0],
        }
    )
    res = build(engine, ScriptedGenerator([{"document": "memorised"}])).run()
    assert res.iterations[0].admitted is True, "the gate only sees the batch, so it admits"
    assert res.best.candidate_id == "c0", "the selector, which never saw the batch, declines it"


def test_the_parent_of_the_second_iteration_can_be_the_seed_again():
    """Reaching back past an admitted candidate is what a population is for."""
    engine = TableEngine(
        table={
            ("c0", "i1"): [0.0],
            ("c0", "i2"): [1.0],
            ("c1", "i1"): [1.0],
            ("c1", "i2"): [1.0],
            ("c0", "s1"): [1.0],
            ("c0", "s2"): [1.0],
            ("c1", "s1"): [1.0],
            ("c1", "s2"): [0.0],
            ("c2", "i1"): [1.0],
            ("c2", "i2"): [1.0],
            ("c2", "s1"): [1.0],
            ("c2", "s2"): [1.0],
        }
    )
    gen = ScriptedGenerator([{"document": "a"}, {"document": "b"}])
    ctl = build(engine, gen, iterations=2, parent=select_by_mean())
    ctl.run()
    assert gen.seen_parents == [
        "c0",
        "c0",
    ], "c0 keeps the better selection mean, so the second proposal extends it again"


def test_iteration_scoped_feedback_keeps_the_gate_comparing_one_batch():
    """A candidate that was a child then becomes a parent must not carry its old batch along."""
    engine = TableEngine(
        table={
            ("c0", "i1"): [0.0],
            ("c0", "i2"): [0.0],
            ("c1", "i1"): [1.0],
            ("c1", "i2"): [1.0],
            ("c0", "s1"): [0.0],
            ("c0", "s2"): [0.0],
            ("c1", "s1"): [1.0],
            ("c1", "s2"): [1.0],
            ("c2", "i1"): [1.0],
            ("c2", "i2"): [1.0],
            ("c2", "s1"): [1.0],
            ("c2", "s2"): [1.0],
        }
    )
    ctl = build(engine, ScriptedGenerator([{"document": "a"}, {"document": "b"}]), iterations=2)
    res = ctl.run()
    assert ctl.store.per_item("c1", "feedback:1") == {"i1": 1.0, "i2": 1.0}
    assert ctl.store.per_item("c1", "feedback:2") == {"i1": 1.0, "i2": 1.0}
    assert (
        res.iterations[1].gate["role"] == "feedback:2"
    ), "the gate must be told which iteration's records to compare"


def test_an_empty_evaluation_raises_rather_than_reading_as_saturated():
    """The recorded failure: zero episodes were read as 'nothing left to solve' and exited 0."""

    class Empty(TableEngine):
        def generate_batch(self, data_samples):
            return iter(())

    with pytest.raises(RuntimeError, match="no rollouts"):
        build(Empty(table={}), ScriptedGenerator([{"document": "x"}])).run()


def test_the_formula_is_materialised_before_each_request():
    formula = DictFormula({"document": ""})
    engine = TableEngine(table={}, default={"c0": [1.0], "c1": [1.0]})
    ctl = build(engine, ScriptedGenerator([{"document": "child"}]), formula=formula)
    ctl.run()
    assert formula.get_tunable_params()["document"] == "child"


def test_a_formula_that_ignores_the_update_is_an_error():
    """Otherwise the engine would silently execute a different candidate than the one requested."""

    class Stubborn(DictFormula):
        def update_params(self, params):
            return None

    engine = TableEngine(table={}, default={"c0": [1.0]})
    with pytest.raises(RuntimeError, match="did not take parameter"):
        build(
            engine, ScriptedGenerator([{"document": "child"}]), formula=Stubborn({"document": ""})
        ).run()


def test_the_view_withholds_the_reference_answer_unless_asked():
    engine = TableEngine(table={}, default={"c0": [1.0], "c1": [1.0]})
    gen = ScriptedGenerator([{"document": "x"}])
    build(engine, gen, view=minimal_view(reveal_reference=False)).run()
    assert gen.seen_views[0].reveals_reference is False

    gen2 = ScriptedGenerator([{"document": "x"}])
    build(
        TableEngine(table={}, default={"c0": [1.0], "c1": [1.0]}),
        gen2,
        view=minimal_view(reveal_reference=True),
    ).run()
    assert gen2.seen_views[0].reveals_reference is True


def test_result_serialises_to_something_reportable():
    engine = TableEngine(table={}, default={"c0": [1.0], "c1": [0.0]})
    res = build(engine, ScriptedGenerator([{"document": "x"}])).run()
    d = res.to_json()
    assert d["best"] == "c0" and d["stop_reason"] and d["rollouts_used"] > 0
    assert d["iterations"][0]["gate"]["basis"] == "own"


def test_evaluate_returns_records_and_the_rollouts_they_came_from():
    """Both halves of an EvaluationPass must be populated.

    The generator reflects on rollouts, not on scores. A controller that returned records alone left
    the teacher with an empty batch — no transcripts, no predictions, no answers — and it did so
    silently, because a reflective proposer still writes something from the rest of its prompt.
    """
    engine = TableEngine(table={}, default={"c0": [1.0]})
    ctl = build(engine, ScriptedGenerator([{"document": "x"}]))
    p = ctl._evaluate(ctl.pool[0], ITEMS, "feedback:1")
    assert p.records and p.rollouts, (len(p.records), len(p.rollouts))
    assert len(p.rollouts) == len(p.records)
    assert {r.item_id for r in p.records} == {
        str((ro.data_sample or {}).get("item_id")) for ro in p.rollouts
    }


def test_feedback_view_is_called_with_the_parent_rollouts():
    """The view must see the rollouts, not only the records, or the proposer gets nothing to read."""
    seen = {}

    def view(rollouts, records):
        seen["rollouts"] = list(rollouts)
        seen["records"] = list(records)
        return list(rollouts)

    engine = TableEngine(table={}, default={"c0": [1.0], "c1": [1.0]})
    ctl = build(engine, ScriptedGenerator([{"document": "x"}]), view=view)
    ctl.run()
    assert seen["rollouts"], "feedback view received no rollouts"
    assert len(seen["rollouts"]) == len(seen["records"])

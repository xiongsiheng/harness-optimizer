# Search

A [FormulaOptimizer](./optimizers.md) commits every edit it proposes: `step()` mutates the Formula in
place and returns nothing. That is what you want for greedy tuning against a dense reward. It is not
enough when an improvement measured on the batch you showed the proposer might not survive a fresh
draw of items — the loop has no point at which an edit can be declined, two edits compared, or the
original kept.

`SearchController` adds that outer layer. Nothing in the existing training path changes.

## The loop

One iteration, in order:

1. **Sample** a batch of feedback items.
2. **Evaluate the parent** on that batch.
3. **Propose** a child from the parent and a view of its rollouts.
4. **Guard** the proposal — a rejected proposal costs no rollouts.
5. **Evaluate the child on the same items.** The comparison is paired by construction.
6. **Gate.** Rejected children are recorded and discarded; the parent stays.
7. **Score admitted candidates** on the selection split, which the proposer never sees.

Stopping is a policy: a rollout budget, or saturation — no items left that the current candidate
fails.

## Minimal use

```python
from strands_harness_optimizer.search import (
    Candidate, SearchController, budget_and_saturation, select_by_mean,
    stratified_sampler, strict_improvement,
)

controller = SearchController(
    seed=Candidate.seed({"system_prompt": "You are a helpful assistant."}),
    generator=my_generator,          # .propose(parent, view) -> params
    engine=my_rollout_engine,
    reward_fn=my_reward_function,
    feedback_items=train_ids,
    selection_items=holdout_ids,
    parent_selection=select_by_mean(),
    item_sampling=stratified_sampler({"never": 10, "sometimes": 10, "always": 4}, seed=0),
    gate=strict_improvement(),
    stopping=budget_and_saturation(rollout_budget=1200, min_unsolved=4),
)

result = controller.run()
result.best.params        # the winning parameters, which may be the seed's
result.iterations         # one record per iteration: gate means, admission, cost
```

`generator` is anything with `propose(parent, view) -> Mapping[str, Any]`. An existing
`FormulaOptimizer` is adapted in about thirty lines: hand it a throwaway Formula holding a copy of the
parent's params, let `step()` mutate that, and read the result back.

## The candidate pool

A `Candidate` is an immutable snapshot of a formula's parameters plus its lineage. `child()` derives a
new candidate without touching the parent, and the seed stays in the pool for the whole search — so
"the original was already the best" is an outcome the search can return, not one it has to be rescued
from. A seed with empty parameters is a legitimate member of the pool.

## Per-item evaluation

`EvaluationStore` keeps one record per (candidate, item, replicate) rather than one number per epoch:

```python
store.per_item(candidate_id, role="selection")   # {item_id: mean score}
store.mean(candidate_id, role="selection")
store.pass_at_k(candidate_id, role="selection")  # (value, k)
```

Three things need this and cannot be computed from an average. Paired comparison, which is what tells
a real gain from noise on a small batch. Difficulty buckets, which is how sampling puts unsolved items
in front of the proposer instead of spending rollouts on items already solved. And the distinction
between a candidate that made previously-unsolvable items solvable and one that merely stabilised
items already sometimes solved — those look identical in the mean.

Each record carries a status: `valid`, `timeout`, `execution_error`, `missing_output`. Execution
errors are excluded from scoring by default, so an infrastructure failure is not silently read as the
model getting the answer wrong.

## What the proposer sees

A reflective proposer reads trajectories. `FeedbackView` states which fields of a rollout travel to
it:

```python
from strands_harness_optimizer.search import minimal_view

view = minimal_view(keep=("prompt", "prediction", "messages"))   # reference withheld
view = minimal_view(reveal_reference=True)                       # reference included
```

The default withholds the reference answer. Shown the answers, a capable proposer writes them into
the artifact — a rational response to the objective, not misbehaviour — and the resulting improvement
does not transfer to items it never saw. Withholding is necessary and not sufficient: it cannot stop a
proposer that reads the corpus through a shell tool, and on a task whose answer space is a shared
candidate list, a proposer denied the answers will copy the model's own predictions instead, which are
other items' answers. Pair it with an instruction in your template and a guard:

```python
from strands_harness_optimizer.search import forbid_values

guard = forbid_values(reference_values, min_length=3)   # rejected before it costs rollouts
```

## Testing a search without paying for rollouts

`ReplayRolloutEngine` serves rollouts you have already recorded, keyed by (candidate, role, item), and
fails closed on a key it does not have. Search decisions are then deterministic, so a policy change
can be tested against recorded runs rather than by re-running an agent.

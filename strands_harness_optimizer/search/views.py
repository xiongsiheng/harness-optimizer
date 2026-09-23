"""What the candidate generator is allowed to see.

The upstream reflection optimizer writes the whole data sample — reference answer included — into
files the proposer reads with a shell tool. Measured consequence on the corpus this design comes
from: learned documents contained verbatim dataset answers on five of eight tasks, one opening with a
section titled "Known gene -> answer table (use this first)", and correcting the resulting inflation
moved a headline result from +10.7 to +8.9 macro. The failure was silent for a day; nothing errored
and every reported number improved.

A view therefore exists so that "what the proposer sees" is an explicit object rather than "whatever
happens to be in the rollout", and so that the choice is recorded in the run's artifacts. The default
withholds the reference, because the failure above is silent and a silent failure needs the safe
default. It is not a security boundary: a proposer with shell access can read the corpus directly, and
withholding on its own was measured to be insufficient (see `minimal_view`). Detecting leakage and
restricting tools remain application concerns; this module only makes the decision explicit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from ..datamodels import Rollout
from .evaluation import EvaluationRecord


@dataclass
class FeedbackView:
    """Trajectories and outcomes handed to the generator, with the label channel explicit."""

    items: list[dict] = field(default_factory=list)
    reveals_reference: bool = False
    fields: tuple[str, ...] = ()

    def to_json(self) -> dict:
        return {
            "n_items": len(self.items),
            "reveals_reference": self.reveals_reference,
            "fields": list(self.fields),
        }


def minimal_view(
    reference_key: str = "answer",
    reveal_reference: bool = False,
    keep: Sequence[str] = ("prompt", "prediction", "messages"),
):
    """Build a view carrying inputs, outputs, trajectories and rewards — and, only when asked,
    the reference answer.

    **The default withholds the reference.** Pass `reveal_reference=True` to include it, and expect
    the artifact to contain answer values if you do.

    Withholding is necessary and, on its own, not sufficient — worth stating because the measured
    result is counter-intuitive. Denied the answer, the proposer copied the student's own predictions
    into the artifact, and on a task whose answer space is a shared candidate list those predictions
    are other instances' answers: ten answer values in the produced document against one when the
    answer was visible. Withholding a field from this view also cannot stop a proposer that reads the
    corpus through a shell tool.

    What did work was *telling* the proposer not to write answers down, plus a guard on the proposal
    before it costs any rollouts. Both belong to the application: the first is a template, the second
    a policy the controller calls. This function only decides which fields travel.
    """

    def build(rollouts: Sequence[Rollout], records: Sequence[EvaluationRecord]) -> FeedbackView:
        by_item: dict[str, dict] = {}
        for rec in records:
            e = by_item.setdefault(
                rec.item_id, {"item_id": rec.item_id, "scores": [], "statuses": []}
            )
            e["scores"].append(rec.score)
            e["statuses"].append(rec.status.value)
        for ro in rollouts:
            item = str((ro.data_sample or {}).get("item_id"))
            e = by_item.setdefault(item, {"item_id": item, "scores": [], "statuses": []})
            src = dict(ro.data_sample or {})
            for k in keep:
                if k in src:
                    e[k] = src[k]
            if ro.messages:
                e["messages"] = ro.messages
            if reveal_reference and reference_key in src:
                e[reference_key] = src[reference_key]
        fields = tuple(sorted({k for v in by_item.values() for k in v}))
        return FeedbackView(
            items=list(by_item.values()), reveals_reference=reveal_reference, fields=fields
        )

    return build

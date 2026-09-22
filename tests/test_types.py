"""kev_showdown.types: the request/response models must accept exactly what kev's real API sends and rejects
nothing valid -- test_live_server.py cross-checks this against the actual running model."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from kev_showdown.types import Choice, ChoiceAnswer, Noul, NoulAnswer, SystemOneRequest, SystemOneResponse


def test_request_round_trips_through_json() -> None:
    req = SystemOneRequest(
        state={"turn": 1},
        questions={
            "action": Choice(instructions="pick one", criteria={"a": "first", "b": "second"}),
            "win": Noul(instructions="will we win?"),
        },
    )
    dumped = req.model_dump(mode="json", exclude_none=True)
    assert SystemOneRequest.model_validate(dumped) == req


def test_choice_rejects_empty_criteria() -> None:
    with pytest.raises(ValidationError):
        Choice(instructions="x", criteria={})


def test_response_discriminates_answer_types_by_type_field() -> None:
    payload = {
        "model": "kev-latest",
        "answers": {
            "action": {"type": "choice", "choice": "a", "confidence": 0.5, "probabilities": {"a": 0.6, "b": 0.4}},
            "win": {"type": "noul", "noul": 0.73},
        },
        "usage": {"input_tokens": 10, "output_tokens": 5},
        "latency_ms": 12.3,
    }
    resp = SystemOneResponse.model_validate(payload)
    action = resp.answers["action"]
    win = resp.answers["win"]
    assert isinstance(action, ChoiceAnswer) and action.choice == "a"
    assert isinstance(win, NoulAnswer) and win.noul == pytest.approx(0.73)


@pytest.mark.parametrize("bad_noul", [-0.1, 1.1])
def test_noul_answer_probability_must_be_in_unit_range(bad_noul: float) -> None:
    with pytest.raises(ValidationError):
        NoulAnswer(noul=bad_noul)


def test_response_rejects_unrecognised_answer_type() -> None:
    with pytest.raises(ValidationError):
        SystemOneResponse.model_validate({
            "model": "kev-latest",
            "answers": {"action": {"type": "essay", "text": "not a real answer kind"}},
            "usage": {"input_tokens": 1, "output_tokens": 1},
            "latency_ms": 1.0,
        })

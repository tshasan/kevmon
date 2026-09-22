"""Typed shapes for Kev's `/v1/systemone` wire contract (request and response).

The request side is TypedDicts, not pydantic: `kev_showdown.common.build_request` runs once per battle turn (and,
in `collect.py`, thousands of times with no model latency to hide the cost) building a request from data this
process already computed -- there's no untrusted input on that path to justify runtime validation. TypedDict gives
mypy the same static guarantee at zero runtime cost.

The response side stays pydantic: it validates a real HTTP reply from Kev, which *is* untrusted input, and that's
exactly the boundary where catching a malformed/drifted shape earns its cost. These mirror kev's own `kev/api.py`
(Noul/Choice/Score) and `kev/serve.py`'s response body. They're declared independently rather than imported from
the `kev` package so kev_showdown never has to install kev's torch/transformers just to get an HTTP client's types.
"""
from __future__ import annotations

from typing import Annotated, Any, Literal, TypedDict, Union

from pydantic import BaseModel, Field

# Not a recursive type alias (dict/list left unparameterised): matches kev's own kev/api.py definition, and a
# recursive `dict[str, "JSONContent"]` union blows pydantic's schema-generation recursion limit.
JSONContent = Union[str, int, float, bool, None, dict[str, Any], list[Any]]

MAX_OPTIONS = 255


# --------------------------------------------------------------------------------------------------------- request

class Question(TypedDict, total=False):
    type: Literal["noul", "choice", "score"]
    instructions: JSONContent
    criteria: JSONContent


class SystemOneRequest(TypedDict):
    state: JSONContent
    model: str
    questions: dict[str, Question]


# -------------------------------------------------------------------------------------------------------- response

class NoulAnswer(BaseModel):
    type: Literal["noul"] = "noul"
    noul: float = Field(ge=0.0, le=1.0)


class ChoiceAnswer(BaseModel):
    type: Literal["choice"] = "choice"
    choice: str
    confidence: float = Field(ge=0.0, le=1.0)
    probabilities: dict[str, float]


class ScoreAnswer(BaseModel):
    type: Literal["score"] = "score"
    score: float
    legend: dict[str, str]
    probabilities: dict[str, float]
    confidence: float = Field(ge=0.0, le=1.0)


Answer = Annotated[Union[NoulAnswer, ChoiceAnswer, ScoreAnswer], Field(discriminator="type")]


class Usage(BaseModel):
    input_tokens: int
    output_tokens: int


class SystemOneResponse(BaseModel):
    model: str
    answers: dict[str, Answer]
    usage: Usage
    latency_ms: float

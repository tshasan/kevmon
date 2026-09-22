"""Typed shapes for Kev's `/v1/systemone` wire contract (request and response).

These mirror kev's own `kev/api.py` (Noul/Choice/Score/SystemOneRequest) and `kev/serve.py`'s response body
byte-for-byte. They are declared independently rather than imported from the `kev` package so that kev_showdown
never has to install kev's heavy training/serving dependencies (torch, transformers, ...) just to get types for
an HTTP client. If kev's wire contract changes, `test_types.py` (`--live`) is what will catch the drift.
"""
from __future__ import annotations

from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, Field

# Not a recursive type alias (dict/list left unparameterised): matches kev's own kev/api.py definition, and a
# recursive `dict[str, "JSONContent"]` union blows pydantic's schema-generation recursion limit.
JSONContent = Union[str, int, float, bool, None, dict[str, Any], list[Any]]

MAX_OPTIONS = 255


# --------------------------------------------------------------------------------------------------------- request

class Noul(BaseModel):
    type: Literal["noul"] = "noul"
    instructions: JSONContent
    criteria: dict[str, JSONContent] | None = None


class Choice(BaseModel):
    type: Literal["choice"] = "choice"
    instructions: JSONContent
    criteria: dict[str, JSONContent] = Field(min_length=1, max_length=MAX_OPTIONS)


class Score(BaseModel):
    type: Literal["score"] = "score"
    instructions: JSONContent
    criteria: list[JSONContent] = Field(min_length=2, max_length=MAX_OPTIONS)


Question = Annotated[Union[Noul, Choice, Score], Field(discriminator="type")]


class SystemOneRequest(BaseModel):
    state: JSONContent
    model: str = "kev-latest"
    questions: dict[str, Question] = Field(min_length=1)


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

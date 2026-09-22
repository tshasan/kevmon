"""Structured JSON-lines logging for every Kev call: one line per event, greppable, no external service required.

    log = configure(Path("kev.log.jsonl"))
    emit(log, "kev.decision", battle_tag=battle.battle_tag, turn=battle.turn, pick=pick, confidence=0.82)

Every line is a single JSON object with at least `ts` and `event`; anything else is caller-supplied. This is
deliberately not a metrics/tracing stack (no OpenTelemetry, no collector to run) -- for this project's scale, a
JSONL file you can `jq` is more observability per line of code than a dashboard nobody's watching.
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

LOGGER_NAME = "kev_showdown"


class _JsonLinesFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {"ts": round(time.time(), 3), "level": record.levelname, "event": record.getMessage()}
        fields = getattr(record, "fields", None)
        if isinstance(fields, dict):
            payload.update(fields)
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, sort_keys=True, default=str)


def configure(log_file: str | Path | None = None, level: int = logging.INFO) -> logging.Logger:
    """Wire up the kev_showdown JSON-lines logger. Idempotent: safe to call from every entrypoint."""
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level)
    logger.propagate = False
    if not logger.handlers:
        stream = logging.StreamHandler(sys.stderr)
        stream.setFormatter(_JsonLinesFormatter())
        logger.addHandler(stream)
    if log_file is not None and not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(_JsonLinesFormatter())
        logger.addHandler(file_handler)
    return logger


def emit(logger: logging.Logger, event: str, /, **fields: Any) -> None:
    logger.log(logging.INFO, event, extra={"fields": fields})


def emit_warning(logger: logging.Logger, event: str, /, **fields: Any) -> None:
    logger.log(logging.WARNING, event, extra={"fields": fields})

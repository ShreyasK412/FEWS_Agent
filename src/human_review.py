"""
Human review utilities for FEWS system.

This module provides a minimal JSONL-backed queue for low-confidence
explanations and a store of human-approved overrides that can be used
as authoritative ground truth on subsequent runs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import HUMAN_REVIEW_QUEUE_FILE, HUMAN_REVIEW_APPROVED_FILE
from .exceptions import FEWSException


@dataclass
class HumanReviewedExplanation:
    """Human-reviewed explanation and drivers for a region/IPC phase."""

    region: str
    ipc_phase: int
    geographic_full_name: str
    shocks_human: List[Dict[str, Any]]
    drivers_human: List[str]
    explanation_human: Optional[str]
    sources_human: List[str]
    reviewed_by: Optional[str]
    reviewed_at: str

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HumanReviewedExplanation":
        """Create record from dictionary, with basic validation."""
        try:
            return cls(
                region=data["region"],
                ipc_phase=int(data["ipc_phase"]),
                geographic_full_name=data.get("geographic_full_name", ""),
                shocks_human=list(data.get("shocks_human", [])),
                drivers_human=list(data.get("drivers_human", [])),
                explanation_human=data.get("explanation_human"),
                sources_human=list(data.get("sources_human", [])),
                reviewed_by=data.get("reviewed_by"),
                reviewed_at=data.get("reviewed_at", ""),
            )
        except KeyError as exc:
            raise FEWSException(f"Invalid human review record, missing field: {exc}") from exc


def _append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    """Append a single JSON object as a line to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def enqueue_explanation_for_review(
    *,
    region: str,
    ipc_phase: int,
    geographic_full_name: str,
    data_quality: str,
    shocks_model: List[Dict[str, Any]],
    drivers_model: List[str],
    explanation_model: str,
    sources_model: List[str],
) -> None:
    """
    Enqueue a low-confidence explanation for human review.

    The queue is a JSONL file where each line is a standalone JSON object.
    """
    record: Dict[str, Any] = {
        "region": region,
        "ipc_phase": ipc_phase,
        "geographic_full_name": geographic_full_name,
        "data_quality": data_quality,
        "shocks_model": shocks_model,
        "drivers_model": drivers_model,
        "explanation_model": explanation_model,
        "sources_model": sources_model,
        "status": "pending",
        "created_at": datetime.utcnow().isoformat(),
    }
    _append_jsonl(Path(HUMAN_REVIEW_QUEUE_FILE), record)


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Load all JSON objects from a JSONL file. Returns empty list if missing."""
    if not path.exists():
        return []

    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    records.append(obj)
            except json.JSONDecodeError:
                # Skip malformed lines but do not raise – logging is handled elsewhere.
                continue
    return records


def load_approved_explanations() -> List[HumanReviewedExplanation]:
    """Load all human-approved explanations from the approved JSONL file."""
    path = Path(HUMAN_REVIEW_APPROVED_FILE)
    raw_records = _load_jsonl(path)
    records: List[HumanReviewedExplanation] = []
    for record in raw_records:
        try:
            records.append(HumanReviewedExplanation.from_dict(record))
        except FEWSException:
            # Ignore invalid records; detailed logging can be added by caller if needed.
            continue
    return records


def find_approved_explanation(region: str, ipc_phase: int) -> Optional[HumanReviewedExplanation]:
    """
    Find a human-approved explanation for a given region and IPC phase.

    Matching is case-insensitive on region name and exact on ipc_phase.
    """
    region_lower = region.lower().strip()
    for record in load_approved_explanations():
        if record.ipc_phase != ipc_phase:
            continue
        if record.region.lower().strip() == region_lower:
            return record
    return None


def should_enqueue_for_review(data_quality: str, shocks: List[Dict[str, Any]]) -> bool:
    """
    Decide whether an automatically generated explanation should be sent for human review.

    Heuristic:
    - If data_quality is not "sufficient", always enqueue.
    - Otherwise, if all shocks have confidence "low" or are special types
      ("insufficient_data" / "data_gap"), enqueue.
    """
    if data_quality != "sufficient":
        return True

    if not shocks:
        return True

    all_low_or_gap = True
    for shock in shocks:
        shock_type = str(shock.get("type", "")).lower()
        confidence = str(shock.get("confidence", "")).lower()
        if shock_type not in ("insufficient_data", "data_gap") and confidence in ("high", "medium"):
            all_low_or_gap = False
            break

    return all_low_or_gap



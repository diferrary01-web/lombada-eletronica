"""Tipos de dominio compartilhados pelo pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

BBox = tuple[float, float, float, float]  # x1, y1, x2, y2 em pixels


@dataclass(frozen=True)
class Detection:
    bbox: BBox
    confidence: float
    label: str


@dataclass
class TrackSample:
    """Uma observacao de um veiculo rastreado, ja projetada no plano da via."""

    frame_ts: float
    bbox: BBox
    world_x: float
    world_y: float
    confidence: float


@dataclass
class Track:
    track_id: int
    label: str
    samples: list[TrackSample] = field(default_factory=list)
    last_seen_ts: float = 0.0

    def series(self) -> list[tuple[float, float]]:
        """Pares (t, Y) para o ajuste de velocidade."""
        return [(s.frame_ts, s.world_y) for s in self.samples]


@dataclass(frozen=True)
class PlateRead:
    text: str
    confidence: float
    bbox: BBox | None = None


@dataclass
class Passage:
    """Uma passagem medida na base -- infratora ou nao."""

    camera_id: str
    track_id: int
    captured_at: datetime
    speed_kmh: float
    considered_kmh: float
    limit_kmh: float
    label: str
    plate: PlateRead | None = None
    quality: dict[str, Any] = field(default_factory=dict)
    evidence: dict[str, str] = field(default_factory=dict)

    @property
    def is_violation(self) -> bool:
        return self.considered_kmh > self.limit_kmh

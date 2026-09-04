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
    """Uma leitura de placa, de um motor, sobre um recorte.

    `source` diz qual motor leu e `weight` quanto o voto dele pesa. Os dois
    existem porque a consolidacao e um ENSEMBLE: a mesma passagem rende varias
    leituras, de motores diferentes e de quadros diferentes, e o voto precisa
    saber distinguir "o PP-OCR e o TrOCR concordaram" de "o mesmo motor repetiu
    o mesmo erro seis vezes".
    """

    text: str
    confidence: float
    bbox: BBox | None = None
    source: str = ""
    weight: float = 1.0

    @property
    def vote_weight(self) -> float:
        return max(self.confidence, 1e-6) * max(self.weight, 0.0)


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

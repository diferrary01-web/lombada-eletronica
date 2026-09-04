"""Rastreio de veiculos por associacao em dois estagios.

Por que nao usar direto um ByteTrack de prateleira: a associacao dele parte da
previsao de um filtro de Kalman que precisa de dois ou tres quadros para
travar. Numa lombada, um veiculo a 50 km/h anda ~1,2 m por quadro a 12 fps e
atravessa a base de 8 m em pouco mais de meio segundo -- ou seja, o rastreador
troca o ID justamente nas passagens que mais interessam. O efeito pratico e um
TETO DE VELOCIDADE silencioso: acima de certa velocidade o sistema para de
gerar eventos, e isso nao aparece como erro em lugar nenhum.

Dai os dois estagios:

1. **IoU sobre a caixa PREVISTA.** Com dois quadros ja ha velocidade, e a
   extrapolacao recoloca a caixa em cima da deteccao seguinte mesmo quando as
   caixas cruas nem se tocam.
2. **Distancia entre centros, com portao.** O estagio 1 nao serve no segundo
   quadro de um alvo novo, porque ainda nao ha velocidade para extrapolar --
   e e exatamente ai que o alvo rapido se perde. O estagio 2 casa pelo centro
   mais proximo, limitado a um portao proporcional ao tamanho da caixa e
   exigindo a mesma classe, que e o que impede colar carros distintos.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .models import BBox, Detection


@dataclass
class _Candidate:
    """Estado interno de um alvo em rastreio."""

    track_id: int
    label: str
    bbox: BBox
    last_ts: float
    velocity: tuple[float, float] = (0.0, 0.0)  # px/s do centro da caixa
    hits: int = 1

    def predict(self, ts: float) -> BBox:
        dt = ts - self.last_ts
        if dt <= 0 or self.hits < 2:
            return self.bbox
        dx, dy = self.velocity[0] * dt, self.velocity[1] * dt
        x1, y1, x2, y2 = self.bbox
        return (x1 + dx, y1 + dy, x2 + dx, y2 + dy)

    def update(self, bbox: BBox, ts: float) -> None:
        dt = ts - self.last_ts
        if dt > 1e-6:
            cx_old, cy_old = center(self.bbox)
            cx_new, cy_new = center(bbox)
            self.velocity = ((cx_new - cx_old) / dt, (cy_new - cy_old) / dt)
        self.bbox = bbox
        self.last_ts = ts
        self.hits += 1


@dataclass(frozen=True)
class TrackedObject:
    """Saida do rastreador para um quadro."""

    track_id: int
    bbox: BBox
    label: str
    confidence: float
    hits: int


@dataclass
class IouTracker:
    """Associa deteccoes a alvos entre quadros.

    `max_age_s` e em segundos, nao em quadros: a taxa efetiva de captura varia
    com a carga da GPU, e contar quadros faria o tempo de vida do alvo mudar
    junto -- outra forma de perder ID em movimento rapido.
    """

    iou_threshold: float = 0.25
    max_age_s: float = 1.0
    min_hits: int = 2
    gate_factor: float = 2.0
    _candidates: list[_Candidate] = field(default_factory=list, repr=False)
    _next_id: int = field(default=1, repr=False)

    def update(self, detections: list[Detection], ts: float) -> list[TrackedObject]:
        self._expire(ts)
        matches = self._associate(detections, ts)

        for index, candidate in matches.items():
            candidate.update(detections[index].bbox, ts)

        for index, detection in enumerate(detections):
            if index in matches:
                continue
            candidate = _Candidate(
                track_id=self._next_id,
                label=detection.label,
                bbox=detection.bbox,
                last_ts=ts,
            )
            self._candidates.append(candidate)
            matches[index] = candidate
            self._next_id += 1

        return [
            TrackedObject(
                track_id=matches[i].track_id,
                bbox=detection.bbox,
                label=matches[i].label,
                confidence=detection.confidence,
                hits=matches[i].hits,
            )
            for i, detection in enumerate(detections)
            if matches[i].hits >= self.min_hits
        ]

    def forget(self, track_id: int) -> None:
        """Descarta um alvo ja consumido (passagem medida e fechada)."""
        self._candidates = [c for c in self._candidates if c.track_id != track_id]

    @property
    def active(self) -> int:
        return len(self._candidates)

    # -- associacao -------------------------------------------------------

    def _associate(
        self, detections: list[Detection], ts: float
    ) -> dict[int, _Candidate]:
        available = [
            (candidate, candidate.predict(ts))
            for candidate in self._candidates
            if candidate.last_ts < ts
        ]
        matches: dict[int, _Candidate] = {}
        taken: set[int] = set()

        scored = sorted(
            (
                (iou(predicted, det.bbox), ci, di)
                for ci, (_cand, predicted) in enumerate(available)
                for di, det in enumerate(detections)
            ),
            key=lambda item: item[0],
            reverse=True,
        )
        for score, ci, di in scored:
            if score < self.iou_threshold:
                break
            if ci in taken or di in matches:
                continue
            taken.add(ci)
            matches[di] = available[ci][0]

        gated = sorted(
            (
                (
                    math.dist(center(predicted), center(det.bbox)),
                    ci,
                    di,
                )
                for ci, (cand, predicted) in enumerate(available)
                if ci not in taken
                for di, det in enumerate(detections)
                if di not in matches
                and det.label == cand.label
                and math.dist(center(predicted), center(det.bbox))
                <= self.gate_factor * _span(predicted)
            ),
            key=lambda item: item[0],
        )
        for _distance, ci, di in gated:
            if ci in taken or di in matches:
                continue
            taken.add(ci)
            matches[di] = available[ci][0]

        return matches

    def _expire(self, ts: float) -> None:
        self._candidates = [
            c for c in self._candidates if ts - c.last_ts <= self.max_age_s
        ]


def iou(a: BBox, b: BBox) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def center(bbox: BBox) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _span(bbox: BBox) -> float:
    x1, y1, x2, y2 = bbox
    return max(abs(x2 - x1), abs(y2 - y1))

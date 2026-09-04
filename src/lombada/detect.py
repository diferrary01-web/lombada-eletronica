"""Deteccao de veiculos, com backend plugavel.

O pipeline nao importa nenhuma biblioteca pesada por conta propria: quem
resolve isso e `build_detector`, e so no momento em que o backend escolhido e
construido. Assim os testes do nucleo (geometria, velocidade, votacao de
placa) rodam sem torch, sem CUDA e sem baixar peso nenhum.
"""

from __future__ import annotations

import logging
from typing import Any, Protocol

from .config import DetectorConfig
from .models import Detection

logger = logging.getLogger(__name__)


class VehicleDetector(Protocol):
    """Contrato minimo de um detector."""

    def detect(self, image: Any) -> list[Detection]:
        """Devolve as deteccoes de veiculo de UM quadro BGR."""
        ...


class NullDetector:
    """Nao detecta nada. Serve para subir o pipeline sem modelo instalado."""

    def detect(self, image: Any) -> list[Detection]:  # noqa: ARG002
        return []


class UltralyticsDetector:
    """YOLO via `ultralytics`. Carrega o modelo na primeira chamada."""

    def __init__(self, config: DetectorConfig) -> None:
        self.config = config
        self._model: Any = None
        self._wanted = {c.lower() for c in config.classes}

    def _load(self) -> Any:
        if self._model is None:
            from ultralytics import YOLO

            logger.info(
                "carregando detector %s em %s", self.config.model, self.config.device
            )
            self._model = YOLO(self.config.model)
            self._model.to(self.config.device)
        return self._model

    def detect(self, image: Any) -> list[Detection]:
        model = self._load()
        results = model.predict(
            image,
            conf=self.config.confidence,
            imgsz=self.config.imgsz,
            device=self.config.device,
            verbose=False,
        )
        out: list[Detection] = []
        for result in results:
            names = result.names
            for box in result.boxes:
                label = str(names[int(box.cls)]).lower()
                if self._wanted and label not in self._wanted:
                    continue
                x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])
                out.append(
                    Detection(
                        bbox=(x1, y1, x2, y2),
                        confidence=float(box.conf),
                        label=label,
                    )
                )
        return out


_BACKENDS: dict[str, Any] = {
    "ultralytics": UltralyticsDetector,
    "stub": lambda _config: NullDetector(),
}


def build_detector(config: DetectorConfig) -> VehicleDetector:
    try:
        factory = _BACKENDS[config.backend]
    except KeyError:
        raise ValueError(
            f"backend de deteccao desconhecido: {config.backend!r} "
            f"(disponiveis: {sorted(_BACKENDS)})"
        ) from None
    return factory(config)

"""Registro da evidencia de uma passagem.

Uma medida de velocidade sem imagem nao vale discussao nenhuma: o primeiro
questionamento que aparece e "nao fui eu" ou "nao estava nessa velocidade".
O que se grava aqui e o suficiente para responder aos dois -- o quadro com a
base de medicao desenhada, o recorte da placa, e um manifesto com o SHA-256 de
cada arquivo, para que se possa mostrar depois que a imagem nao foi editada.

O manifesto tambem guarda os parametros da medida (R^2, numero de amostras,
extrapolacao). Sem eles a velocidade e um numero sem procedencia.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import CameraConfig
from .models import Passage

logger = logging.getLogger(__name__)

_JPEG_QUALITY = 92


class EvidenceWriter:
    """Escreve os arquivos de evidencia numa pasta por camera e por dia."""

    def __init__(self, base_dir: str | Path) -> None:
        self.base_dir = Path(base_dir)

    def write(
        self,
        passage: Passage,
        camera: CameraConfig,
        *,
        overview: Any = None,
        plate_crop: Any = None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, str]:
        """Grava os artefatos e devolve os caminhos relativos a `base_dir`."""
        folder = self._folder(passage)
        folder.mkdir(parents=True, exist_ok=True)
        stem = f"{passage.captured_at.strftime('%H%M%S')}_{passage.track_id}"

        written: dict[str, str] = {}
        if overview is not None:
            path = folder / f"{stem}_overview.jpg"
            if self._imwrite(path, overview):
                written["overview"] = self._relative(path)
        if plate_crop is not None and getattr(plate_crop, "size", 1) > 0:
            path = folder / f"{stem}_plate.jpg"
            if self._imwrite(path, plate_crop):
                written["plate"] = self._relative(path)

        manifest_path = folder / f"{stem}_manifest.json"
        manifest = self._manifest(passage, camera, written, extra or {})
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, default=_fallback),
            encoding="utf-8",
        )
        written["manifest"] = self._relative(manifest_path)
        return written

    def _folder(self, passage: Passage) -> Path:
        day = passage.captured_at.strftime("%Y-%m-%d")
        return self.base_dir / passage.camera_id / day

    def _relative(self, path: Path) -> str:
        return path.relative_to(self.base_dir).as_posix()

    def _manifest(
        self,
        passage: Passage,
        camera: CameraConfig,
        files: dict[str, str],
        extra: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "gerado_em": datetime.now().astimezone().isoformat(),
            "camera": {
                "id": camera.id,
                "nome": camera.name,
                "limite_kmh": camera.limit_kmh,
                "tolerancia_kmh": camera.tolerance_kmh,
                "base": {
                    "distancia_m": camera.base.distance_m,
                    "pontos_imagem": [list(p) for p in camera.base.image_points],
                },
            },
            "medida": {
                "instante": passage.captured_at.isoformat(),
                "velocidade_medida_kmh": round(passage.speed_kmh, 2),
                "velocidade_considerada_kmh": round(passage.considered_kmh, 2),
                "infracao": passage.is_violation,
                "classe": passage.label,
                "track_id": passage.track_id,
                "qualidade": passage.quality,
            },
            "placa": (
                {
                    "texto": passage.plate.text,
                    "confianca": round(passage.plate.confidence, 4),
                    # Quais motores leram: uma placa lida por dois motores
                    # independentes se defende melhor que uma lida por um.
                    "motores": passage.plate.source,
                }
                if passage.plate
                else None
            ),
            "arquivos": {
                nome: {
                    "caminho": rel,
                    "sha256": sha256_file(self.base_dir / rel),
                }
                for nome, rel in files.items()
            },
            "extra": extra,
        }

    @staticmethod
    def _imwrite(path: Path, image: Any) -> bool:
        try:
            import cv2

            return bool(
                cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, _JPEG_QUALITY])
            )
        except Exception as exc:  # imagem invalida, disco cheio, sem cv2
            logger.error("falha ao gravar %s: %s", path, exc)
            return False


def draw_overview(image: Any, passage: Passage, camera: CameraConfig) -> Any:
    """Desenha a base de medicao e o resultado sobre uma copia do quadro."""
    import cv2
    import numpy as np

    canvas = image.copy()
    points = np.array(camera.base.image_points, dtype=np.int32).reshape((-1, 1, 2))
    color = (0, 0, 255) if passage.is_violation else (0, 200, 0)
    cv2.polylines(canvas, [points], isClosed=True, color=color, thickness=2)

    lines = [
        f"{camera.name}  limite {camera.limit_kmh:.0f} km/h",
        f"{passage.speed_kmh:.1f} km/h medidos"
        f"  ({passage.considered_kmh:.1f} considerados)",
        passage.captured_at.strftime("%d/%m/%Y %H:%M:%S"),
    ]
    if passage.plate:
        lines.append(f"placa {passage.plate.text} ({passage.plate.confidence:.0%})")

    y = 30
    for line in lines:
        cv2.putText(
            canvas, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4
        )
        cv2.putText(
            canvas, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1
        )
        y += 30
    return canvas


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _fallback(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if isinstance(value, Path):
        return value.as_posix()
    return str(value)

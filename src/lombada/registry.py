"""Cadastro de cameras gravado no MESMO `cameras.yaml` que a CLI le.

A tela web nao mantem banco proprio de cameras de proposito: ela edita o
arquivo de configuracao de verdade. Assim o que voce cadastra no navegador e
exatamente o que `lombada check` valida e `lombada run` executa -- sem um
segundo cadastro para divergir do primeiro.

Uma camera recem-cadastrada entra **desabilitada**, porque ainda nao tem os
quatro pontos da base. Sem calibracao nao existe medida de velocidade, so
video; deixa-la ligada daria a impressao de um sistema funcionando que na
verdade nao mede nada. Ela liga sozinha quando a calibracao e salva.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .probe import mask_url

# Quadrilatero neutro gravado enquanto a camera nao foi calibrada: o
# `cameras.yaml` precisa de 4 pontos para ser carregavel, mas a camera fica
# desabilitada, entao esses numeros nunca chegam a medir nada.
PLACEHOLDER_POINTS: list[list[float]] = [
    [100.0, 700.0],
    [1100.0, 700.0],
    [900.0, 400.0],
    [300.0, 400.0],
]

DEFAULT_SECTIONS: dict[str, Any] = {
    "site": {"name": "Local de teste", "timezone": "America/Sao_Paulo"},
    "detector": {"backend": "stub"},
    "lpr": {"enabled": False},
    "storage": {
        "database_path": "data/lombada.db",
        "evidence_dir": "data/evidence",
        "retention_days": 30,
        "store_non_violations": True,
    },
}


class RegistryError(ValueError):
    """Cadastro invalido."""


@dataclass
class CameraDraft:
    """Uma camera cadastrada, calibrada ou nao."""

    id: str
    name: str
    rtsp_url: str
    substream_url: str | None = None
    limit_kmh: float = 30.0
    distance_m: float = 8.0
    lane_width_m: float = 3.5
    capture_fps: float = 12.0
    image_points: list[list[float]] | None = None
    enabled: bool = False

    @property
    def calibrated(self) -> bool:
        return bool(self.image_points) and len(self.image_points or []) == 4

    def to_entry(self) -> dict[str, Any]:
        """Como esta camera e gravada no `cameras.yaml`."""
        return {
            "id": self.id,
            "name": self.name,
            "enabled": self.enabled and self.calibrated,
            "calibrada": self.calibrated,
            "rtsp_url": self.rtsp_url,
            **({"substream_url": self.substream_url} if self.substream_url else {}),
            "limit_kmh": self.limit_kmh,
            "capture_fps": self.capture_fps,
            "base": {
                "distance_m": self.distance_m,
                "lane_width_m": self.lane_width_m,
                "image_points": self.image_points or PLACEHOLDER_POINTS,
            },
        }

    def to_public(self) -> dict[str, Any]:
        """Como esta camera e devolvida ao navegador -- sem senha."""
        return {
            "id": self.id,
            "name": self.name,
            "enabled": self.enabled and self.calibrated,
            "calibrada": self.calibrated,
            "rtsp_url": mask_url(self.rtsp_url),
            "substream_url": mask_url(self.substream_url or "") or None,
            "limit_kmh": self.limit_kmh,
            "distance_m": self.distance_m,
            "lane_width_m": self.lane_width_m,
            "capture_fps": self.capture_fps,
            "image_points": self.image_points,
        }

    @classmethod
    def from_entry(cls, entry: dict[str, Any]) -> CameraDraft:
        base = entry.get("base") or {}
        points = base.get("image_points")
        calibrated = bool(entry.get("calibrada", True))
        return cls(
            id=str(entry["id"]),
            name=str(entry.get("name", entry["id"])),
            rtsp_url=str(entry.get("rtsp_url", "")),
            substream_url=entry.get("substream_url") or None,
            limit_kmh=float(entry.get("limit_kmh", 30.0)),
            distance_m=float(base.get("distance_m", 8.0)),
            lane_width_m=float(base.get("lane_width_m", 3.5)),
            capture_fps=float(entry.get("capture_fps", 12.0)),
            image_points=(
                [[float(p[0]), float(p[1])] for p in points]
                if points and calibrated
                else None
            ),
            enabled=bool(entry.get("enabled", False)),
        )

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> CameraDraft:
        """Valida o que veio do formulario."""
        rtsp = str(payload.get("rtsp_url", "")).strip()
        if not rtsp:
            raise RegistryError("informe a URL RTSP")
        if not rtsp.lower().startswith(("rtsp://", "rtsps://", "http://", "https://")):
            raise RegistryError(
                "a URL precisa comecar com rtsp:// (ou http:// para MJPEG)"
            )

        name = str(payload.get("name", "")).strip() or "Camera sem nome"
        camera_id = str(payload.get("id", "")).strip() or slugify(name)

        draft = cls(
            id=camera_id,
            name=name,
            rtsp_url=rtsp,
            substream_url=(str(payload.get("substream_url", "")).strip() or None),
            limit_kmh=_positive(payload.get("limit_kmh"), 30.0, "limite de velocidade"),
            distance_m=_positive(payload.get("distance_m"), 8.0, "distancia da base"),
            lane_width_m=_positive(payload.get("lane_width_m"), 3.5, "largura da faixa"),
            capture_fps=_positive(payload.get("capture_fps"), 12.0, "fps de captura"),
            image_points=_points(payload.get("image_points")),
            enabled=bool(payload.get("enabled", True)),
        )
        return draft


@dataclass
class CameraRegistry:
    """Le e grava o `cameras.yaml`, preservando as demais secoes."""

    path: Path
    _document: dict[str, Any] = field(default_factory=dict, repr=False)

    def load(self) -> dict[str, Any]:
        if self.path.is_file():
            loaded = yaml.safe_load(self.path.read_text(encoding="utf-8")) or {}
            if not isinstance(loaded, dict):
                raise RegistryError("o cameras.yaml nao contem um mapeamento")
            self._document = loaded
        else:
            self._document = {**DEFAULT_SECTIONS, "cameras": []}
        self._document.setdefault("cameras", [])
        return self._document

    def cameras(self) -> list[CameraDraft]:
        document = self.load()
        drafts = []
        for entry in document.get("cameras") or []:
            if isinstance(entry, dict) and entry.get("id"):
                drafts.append(CameraDraft.from_entry(entry))
        return drafts

    def get(self, camera_id: str) -> CameraDraft | None:
        for draft in self.cameras():
            if draft.id == camera_id:
                return draft
        return None

    def save(self, draft: CameraDraft) -> CameraDraft:
        document = self.load()
        entries = [
            entry
            for entry in (document.get("cameras") or [])
            if isinstance(entry, dict) and entry.get("id") != draft.id
        ]
        entries.append(draft.to_entry())
        document["cameras"] = entries
        self._write(document)
        return draft

    def delete(self, camera_id: str) -> bool:
        document = self.load()
        entries = document.get("cameras") or []
        remaining = [
            entry
            for entry in entries
            if isinstance(entry, dict) and entry.get("id") != camera_id
        ]
        if len(remaining) == len(entries):
            return False
        document["cameras"] = remaining
        self._write(document)
        return True

    def calibrate(self, camera_id: str, points: Any) -> CameraDraft:
        """Grava os 4 pontos e liga a camera."""
        draft = self.get(camera_id)
        if draft is None:
            raise RegistryError(f"camera desconhecida: {camera_id}")
        parsed = _points(points)
        if parsed is None:
            raise RegistryError("sao necessarios exatamente 4 pontos")
        draft.image_points = parsed
        draft.enabled = True
        return self.save(draft)

    def _write(self, document: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        text = yaml.safe_dump(
            document, allow_unicode=True, sort_keys=False, default_flow_style=False
        )
        header = (
            "# Gerado pela tela web do lombada-eletronica.\n"
            "# Este arquivo contem credencial de camera em texto puro --\n"
            "# ele esta no .gitignore e deve continuar fora do git.\n"
        )
        self.path.write_text(header + text, encoding="utf-8")


def slugify(value: str) -> str:
    """Gera um id seguro para caminho de arquivo e segmento de URL.

    Acentos sao removidos de proposito: o id vira nome de diretorio em
    `evidence/<id>/<dia>/` e parte da URL da API. Manter `í` ali funciona ate
    o dia em que nao funciona -- outro sistema de arquivos, outro locale, um
    zip, um cliente HTTP menos tolerante.
    """
    sem_acento = (
        unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    )
    slug = "".join(c.lower() if c.isalnum() else "-" for c in sem_acento.strip())
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug.strip("-") or "camera"


def _positive(value: Any, default: float, label: str) -> float:
    if value in (None, ""):
        return default
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise RegistryError(f"{label}: valor invalido ({value!r})") from None
    if number <= 0:
        raise RegistryError(f"{label}: precisa ser maior que zero")
    return number


def _points(value: Any) -> list[list[float]] | None:
    if not value:
        return None
    try:
        points = [[float(p[0]), float(p[1])] for p in value]
    except (TypeError, ValueError, IndexError):
        raise RegistryError("pontos de calibracao malformados") from None
    return points if len(points) == 4 else None

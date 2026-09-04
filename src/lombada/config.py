"""Carga e validacao da configuracao (YAML + variaveis de ambiente).

Credenciais NUNCA ficam no YAML: qualquer campo de texto aceita `${VAR}`, que
e resolvido a partir do ambiente na carga. O arquivo de exemplo usa isso nas
URLs RTSP, e por isso pode ser versionado sem vazar senha de camera.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from datetime import time
from pathlib import Path
from typing import Any

import yaml

from .geometry import BaseGeometry, CalibrationError

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


class ConfigError(ValueError):
    """Configuracao ausente, mal formada ou incoerente."""


@dataclass(frozen=True)
class Schedule:
    """Janela diaria de operacao. Cruza a meia-noite quando start > end."""

    start: time
    end: time

    def contains(self, moment: time) -> bool:
        if self.start <= self.end:
            return self.start <= moment <= self.end
        return moment >= self.start or moment <= self.end

    @classmethod
    def parse(cls, raw: dict[str, Any] | None) -> Schedule | None:
        if not raw:
            return None
        try:
            return cls(_parse_time(raw["start"]), _parse_time(raw["end"]))
        except KeyError as exc:
            raise ConfigError(f"schedule sem campo {exc}") from exc


@dataclass(frozen=True)
class CameraConfig:
    id: str
    name: str
    rtsp_url: str
    base: BaseGeometry
    limit_kmh: float
    enabled: bool = True
    substream_url: str | None = None
    schedule: Schedule | None = None
    capture_fps: float = 12.0
    tolerance_kmh: float = 7.0

    def active_at(self, moment: time) -> bool:
        if not self.enabled:
            return False
        return self.schedule is None or self.schedule.contains(moment)


@dataclass(frozen=True)
class DetectorConfig:
    backend: str = "ultralytics"
    model: str = "yolo11n.pt"
    device: str = "cuda:0"
    confidence: float = 0.35
    imgsz: int = 640
    classes: tuple[str, ...] = ("car", "motorcycle", "bus", "truck")


@dataclass(frozen=True)
class LprConfig:
    enabled: bool = True
    backend: str = "fast_plate_ocr"
    detector_model: str = "yolo-v9-t-256-license-plate-end2end"
    recognizer_model: str = "global-plates-mobile-vit-v2-model"
    device: str = "cuda:0"
    min_confidence: float = 0.45
    frames_per_passage: int = 6


@dataclass(frozen=True)
class StorageConfig:
    database_path: Path = Path("data/lombada.db")
    evidence_dir: Path = Path("data/evidence")
    retention_days: int = 30
    store_non_violations: bool = False


@dataclass(frozen=True)
class QualityConfig:
    min_samples: int = 4
    min_r2: float = 0.90
    max_extrapolation_m: float = 2.0
    max_speed_kmh: float = 250.0


@dataclass(frozen=True)
class AppConfig:
    site_name: str = "sem-nome"
    timezone: str = "America/Sao_Paulo"
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    lpr: LprConfig = field(default_factory=LprConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    cameras: tuple[CameraConfig, ...] = ()

    def camera(self, camera_id: str) -> CameraConfig:
        for cam in self.cameras:
            if cam.id == camera_id:
                return cam
        raise ConfigError(f"camera desconhecida: {camera_id}")


def load_config(path: str | Path) -> AppConfig:
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"arquivo de configuracao nao encontrado: {path}")

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ConfigError("a raiz do YAML precisa ser um mapeamento")
    raw = _expand_env(raw)

    site = raw.get("site") or {}
    entries = raw.get("cameras") or []
    cameras = tuple(_camera(entry, i) for i, entry in enumerate(entries))
    if not cameras:
        raise ConfigError("nenhuma camera configurada")

    ids = [c.id for c in cameras]
    duplicates = {i for i in ids if ids.count(i) > 1}
    if duplicates:
        raise ConfigError(f"ids de camera repetidos: {sorted(duplicates)}")

    return AppConfig(
        site_name=str(site.get("name", "sem-nome")),
        timezone=str(site.get("timezone", "America/Sao_Paulo")),
        detector=_build(DetectorConfig, raw.get("detector"), tuples={"classes"}),
        lpr=_build(LprConfig, raw.get("lpr")),
        storage=_storage(raw.get("storage")),
        quality=_build(QualityConfig, raw.get("quality")),
        cameras=cameras,
    )


def _camera(entry: Any, index: int) -> CameraConfig:
    if not isinstance(entry, dict):
        raise ConfigError(f"camera na posicao {index} nao e um mapeamento")
    try:
        cam_id = str(entry["id"])
        base_raw = entry["base"]
        points = base_raw["image_points"]
        rtsp = str(entry["rtsp_url"])
    except KeyError as exc:
        raise ConfigError(
            f"camera na posicao {index} sem campo obrigatorio {exc}"
        ) from exc

    if len(points) != 4:
        raise ConfigError(f"camera {cam_id}: image_points precisa ter 4 pontos")

    corners = tuple((float(p[0]), float(p[1])) for p in points)
    try:
        base = BaseGeometry(
            image_points=corners,  # type: ignore[arg-type]
            distance_m=float(base_raw.get("distance_m", 8.0)),
            lane_width_m=float(base_raw.get("lane_width_m", 3.5)),
        )
        base.homography()  # falha cedo se a calibracao for degenerada
    except CalibrationError as exc:
        raise ConfigError(f"camera {cam_id}: calibracao invalida -- {exc}") from exc

    return CameraConfig(
        id=cam_id,
        name=str(entry.get("name", cam_id)),
        rtsp_url=rtsp,
        substream_url=_opt_str(entry.get("substream_url")),
        base=base,
        limit_kmh=float(entry.get("limit_kmh", 30.0)),
        enabled=bool(entry.get("enabled", True)),
        schedule=Schedule.parse(entry.get("schedule")),
        capture_fps=float(entry.get("capture_fps", 12.0)),
        tolerance_kmh=float(entry.get("tolerance_kmh", 7.0)),
    )


def _storage(raw: Any) -> StorageConfig:
    raw = raw or {}
    return StorageConfig(
        database_path=Path(str(raw.get("database_path", "data/lombada.db"))),
        evidence_dir=Path(str(raw.get("evidence_dir", "data/evidence"))),
        retention_days=int(raw.get("retention_days", 30)),
        store_non_violations=bool(raw.get("store_non_violations", False)),
    )


def _build(cls: type, raw: Any, tuples: set[str] | None = None) -> Any:
    raw = raw or {}
    if not isinstance(raw, dict):
        raise ConfigError(f"secao {cls.__name__} precisa ser um mapeamento")
    known = set(cls.__dataclass_fields__)  # type: ignore[attr-defined]
    unknown = set(raw) - known
    if unknown:
        raise ConfigError(f"{cls.__name__}: chaves desconhecidas {sorted(unknown)}")
    kwargs = dict(raw)
    for key in tuples or ():
        if key in kwargs:
            kwargs[key] = tuple(kwargs[key])
    return cls(**kwargs)


def _opt_str(value: Any) -> str | None:
    return None if value is None else str(value)


def _parse_time(value: Any) -> time:
    if isinstance(value, time):
        return value
    text = str(value).strip()
    parts = text.split(":")
    if len(parts) not in (2, 3):
        raise ConfigError(f"horario invalido: {value!r} (esperado HH:MM)")
    try:
        numbers = [int(part) for part in parts]
        second = numbers[2] if len(numbers) == 3 else 0
        return time(numbers[0], numbers[1], second)
    except ValueError as exc:
        raise ConfigError(f"horario invalido: {value!r}") from exc


def _expand_env(node: Any) -> Any:
    """Resolve `${VAR}` e `${VAR:-default}` recursivamente em strings."""
    if isinstance(node, dict):
        return {k: _expand_env(v) for k, v in node.items()}
    if isinstance(node, list):
        return [_expand_env(v) for v in node]
    if isinstance(node, str):
        return _ENV_PATTERN.sub(_replace_env, node)
    return node


def _replace_env(match: re.Match[str]) -> str:
    name, default = match.group(1), match.group(2)
    value = os.environ.get(name)
    if value is not None:
        return value
    if default is not None:
        return default
    raise ConfigError(
        f"variavel de ambiente {name} referenciada na configuracao mas nao definida"
    )

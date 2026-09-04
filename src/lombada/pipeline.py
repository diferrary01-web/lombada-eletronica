"""Orquestracao por camera: quadro -> deteccao -> rastreio -> medida -> registro.

O laco fecha uma passagem quando o alvo SAI da base ou some por tempo demais,
e so entao gasta OCR. Ler placa dentro do laco de captura e o erro classico
deste tipo de sistema: o OCR passa a ditar a taxa de quadros, a taxa cai, o
rastreio perde ID em velocidade alta e o sistema deixa de ver exatamente as
passagens que deveria pegar. Aqui o laco quente so detecta e rastreia; o OCR
roda uma vez por passagem, sobre os recortes ja guardados.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from .capture import Frame, RtspCapture
from .config import AppConfig, CameraConfig
from .detect import VehicleDetector, build_detector
from .evidence import EvidenceWriter, draw_overview
from .geometry import CalibrationError, contact_point, image_to_world
from .lpr import PlateReader, build_reader, crop_bbox, is_plausible, vote_plates
from .models import BBox, Passage, PlateRead, Track, TrackSample
from .speed import SpeedError, apply_legal_tolerance, fit_speed
from .storage import PassageStore
from .track import IouTracker

logger = logging.getLogger(__name__)

# Folga, em metros, alem da base onde o alvo ainda e considerado na medida.
APPROACH_MARGIN_M = 4.0
# Folga lateral, em multiplos da meia-largura declarada da faixa.
LATERAL_MARGIN = 2.0
# Tempo sem atualizacao apos o qual a passagem e fechada.
CLOSE_AFTER_S = 1.5


@dataclass
class _Pending:
    """Uma passagem em andamento, acumulando amostras e recortes."""

    track: Track
    crops: list[tuple[float, Any]] = field(default_factory=list)
    overview: tuple[float, Any] | None = None
    wall_offset: float = 0.0

    def note_crop(self, image: Any, bbox: BBox, keep: int) -> None:
        crop = crop_bbox(image, bbox)
        if getattr(crop, "size", 0) == 0:
            return
        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        self.crops.append((area, crop))
        # Os maiores recortes sao os mais proximos da camera -- e os que o OCR
        # de fato consegue ler.
        self.crops.sort(key=lambda item: item[0], reverse=True)
        del self.crops[keep:]

    def note_overview(self, image: Any, world_y: float, mid_y: float) -> None:
        distance = abs(world_y - mid_y)
        if self.overview is None or distance < self.overview[0]:
            self.overview = (distance, image.copy())


class CameraPipeline:
    """Roda uma camera de ponta a ponta."""

    def __init__(
        self,
        camera: CameraConfig,
        config: AppConfig,
        *,
        detector: VehicleDetector | None = None,
        reader: PlateReader | None = None,
        store: PassageStore | None = None,
        evidence: EvidenceWriter | None = None,
    ) -> None:
        self.camera = camera
        self.config = config
        self.detector = detector or build_detector(config.detector)
        self.reader = reader or build_reader(config.lpr)
        self.store = store or PassageStore(config.storage.database_path)
        self.evidence = evidence or EvidenceWriter(config.storage.evidence_dir)
        self.homography = camera.base.homography()
        self.tracker = IouTracker()
        self._pending: dict[int, _Pending] = {}
        self.measured = 0
        self.violations = 0

    # -- laco principal ---------------------------------------------------

    def run(self, stop: threading.Event) -> None:
        source = self.camera.substream_url or self.camera.rtsp_url
        capture = RtspCapture(
            source, target_fps=self.camera.capture_fps, name=self.camera.id
        )
        with capture:
            for frame in capture.frames():
                if stop.is_set():
                    break
                if not self.camera.active_at(
                    datetime.fromtimestamp(frame.wall_ts).time()
                ):
                    continue
                try:
                    self.process(frame)
                except Exception:  # uma camera nao pode derrubar as outras
                    logger.exception("%s: erro ao processar quadro", self.camera.id)
        self.flush()

    def process(self, frame: Frame) -> None:
        """Consome um quadro: atualiza alvos e fecha o que saiu da base."""
        detections = self.detector.detect(frame.image)
        tracked = self.tracker.update(detections, frame.monotonic_ts)

        seen: set[int] = set()
        for obj in tracked:
            world = self._project(obj.bbox)
            if world is None:
                continue
            world_x, world_y = world
            if not self._inside_base(world_x, world_y):
                continue

            seen.add(obj.track_id)
            pending = self._pending.get(obj.track_id)
            if pending is None:
                pending = _Pending(
                    track=Track(track_id=obj.track_id, label=obj.label),
                    wall_offset=frame.wall_ts - frame.monotonic_ts,
                )
                self._pending[obj.track_id] = pending

            pending.track.samples.append(
                TrackSample(
                    frame_ts=frame.monotonic_ts,
                    bbox=obj.bbox,
                    world_x=world_x,
                    world_y=world_y,
                    confidence=obj.confidence,
                )
            )
            pending.track.last_seen_ts = frame.monotonic_ts
            pending.note_crop(frame.image, obj.bbox, self.config.lpr.frames_per_passage)
            pending.note_overview(
                frame.image, world_y, self.camera.base.distance_m / 2.0
            )

        self._close_finished(frame.monotonic_ts, seen)

    def flush(self) -> None:
        """Fecha tudo que ficou aberto (parada do servico)."""
        for track_id in list(self._pending):
            self._finish(track_id)

    # -- interno ----------------------------------------------------------

    def _project(self, bbox: BBox) -> tuple[float, float] | None:
        try:
            return image_to_world(self.homography, contact_point(bbox))
        except CalibrationError:
            return None

    def _inside_base(self, world_x: float, world_y: float) -> bool:
        half = self.camera.base.lane_width_m / 2.0 * LATERAL_MARGIN
        if abs(world_x) > half:
            return False
        return (
            -APPROACH_MARGIN_M
            <= world_y
            <= self.camera.base.distance_m + APPROACH_MARGIN_M
        )

    def _close_finished(self, now: float, seen: set[int]) -> None:
        for track_id, pending in list(self._pending.items()):
            if track_id in seen:
                continue
            if now - pending.track.last_seen_ts >= CLOSE_AFTER_S:
                self._finish(track_id)

    def _finish(self, track_id: int) -> None:
        pending = self._pending.pop(track_id, None)
        if pending is None:
            return
        self.tracker.forget(track_id)

        quality = self.config.quality
        try:
            estimate = fit_speed(
                pending.track.series(),
                self.camera.base.distance_m,
                min_samples=quality.min_samples,
                min_r2=quality.min_r2,
                max_extrapolation_m=quality.max_extrapolation_m,
                max_speed_kmh=quality.max_speed_kmh,
            )
        except SpeedError as exc:
            logger.debug("%s: passagem %d descartada: %s", self.camera.id, track_id, exc)
            return

        considered = apply_legal_tolerance(
            estimate.speed_kmh,
            self.camera.limit_kmh,
            tolerance_kmh=self.camera.tolerance_kmh,
        )
        is_violation = considered > self.camera.limit_kmh
        self.measured += 1
        if is_violation:
            self.violations += 1

        if not is_violation and not self.config.storage.store_non_violations:
            return

        passage = Passage(
            camera_id=self.camera.id,
            track_id=track_id,
            captured_at=datetime.fromtimestamp(estimate.t_entry + pending.wall_offset),
            speed_kmh=estimate.speed_kmh,
            considered_kmh=considered,
            limit_kmh=self.camera.limit_kmh,
            label=pending.track.label,
            plate=self._read_plate(pending),
            quality={
                "r2": round(estimate.r2, 4),
                "modelo": estimate.model,
                "amostras": estimate.n_samples,
                "residuo_m": round(estimate.residual_std_m, 4),
                "extrapolado_m": round(estimate.extrapolated_m, 3),
                "dt_s": round(estimate.dt, 4),
                "sentido_invertido": estimate.reversed_direction,
            },
        )

        overview = pending.overview[1] if pending.overview else None
        if overview is not None:
            try:
                overview = draw_overview(overview, passage, self.camera)
            except Exception:
                logger.exception("%s: falha ao anotar o quadro", self.camera.id)

        plate_crop = pending.crops[0][1] if pending.crops else None
        passage.evidence = self.evidence.write(
            passage, self.camera, overview=overview, plate_crop=plate_crop
        )
        self.store.save(passage)
        logger.info(
            "%s: %.1f km/h (%.1f considerados, limite %.0f) placa=%s R2=%.3f",
            self.camera.id,
            passage.speed_kmh,
            passage.considered_kmh,
            passage.limit_kmh,
            passage.plate.text if passage.plate else "-",
            estimate.r2,
        )

    def _read_plate(self, pending: _Pending) -> PlateRead | None:
        reads: list[PlateRead] = []
        for _area, crop in pending.crops:
            try:
                read = self.reader.read(crop)
            except Exception:
                logger.exception("%s: erro no LPR", self.camera.id)
                continue
            if read is not None:
                reads.append(read)
        if not reads:
            return None
        voted = vote_plates(reads)
        if voted and not is_plausible(voted.text):
            logger.debug("%s: placa fora de formato: %s", self.camera.id, voted.text)
        return voted


def run_all(config: AppConfig, stop: threading.Event) -> list[CameraPipeline]:
    """Sobe uma thread por camera habilitada e espera todas terminarem."""
    store = PassageStore(config.storage.database_path)
    evidence = EvidenceWriter(config.storage.evidence_dir)
    detector = build_detector(config.detector)
    reader = build_reader(config.lpr)

    pipelines = [
        CameraPipeline(
            cam,
            config,
            detector=detector,
            reader=reader,
            store=store,
            evidence=evidence,
        )
        for cam in config.cameras
        if cam.enabled
    ]
    if not pipelines:
        logger.warning("nenhuma camera habilitada")
        return []

    threads = [
        threading.Thread(target=p.run, args=(stop,), name=f"pipe:{p.camera.id}")
        for p in pipelines
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return pipelines

"""Leitura de RTSP em thread propria, com reconexao e carimbo de tempo proprio.

Tres cuidados que a leitura ingenua com `cv2.VideoCapture.read()` num laco
nao tem, e que aparecem em campo:

1. **`grab()` continuo, `retrieve()` na taxa alvo.** Se o consumidor decodifica
   mais devagar que a camera emite, o buffer do FFmpeg enche e os quadros
   entregues ficam cada vez mais atrasados. `grab()` e barato (so descarta o
   pacote); decodificar so o quadro que sera usado mantem a latencia estavel.
2. **Timestamp no momento da captura.** O relogio da camera (OSD) pode estar
   dezenas de segundos fora do real, e `CAP_PROP_POS_MSEC` nao e confiavel em
   fluxo ao vivo. O tempo que entra na medida e o monotonico do host no
   instante do `grab()` -- e monotonico, nao `time.time()`, porque um ajuste
   de NTP no meio de uma passagem estragaria a velocidade.
3. **Reconexao com espera.** Camera que cai volta sozinha, sem derrubar o
   processo nem entrar em laco quente de reconexao.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Frame:
    """Um quadro decodificado com o instante em que foi capturado."""

    index: int
    monotonic_ts: float
    wall_ts: float
    image: Any  # numpy.ndarray BGR


class CaptureError(RuntimeError):
    """Falha irrecuperavel na abertura do fluxo."""


class RtspCapture:
    """Fonte de quadros de uma camera, na taxa alvo, sempre o mais recente."""

    def __init__(
        self,
        url: str,
        *,
        target_fps: float = 12.0,
        ring_seconds: float = 6.0,
        reconnect_delay: float = 5.0,
        open_timeout_s: float = 15.0,
        name: str = "camera",
    ) -> None:
        if target_fps <= 0:
            raise ValueError("target_fps deve ser positivo")
        self.url = url
        self.target_fps = target_fps
        self.name = name
        self.reconnect_delay = reconnect_delay
        self.open_timeout_s = open_timeout_s
        self._interval = 1.0 / target_fps
        self._ring: deque[Frame] = deque(maxlen=max(1, int(ring_seconds * target_fps)))
        self._lock = threading.Lock()
        self._latest: Frame | None = None
        self._new_frame = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._index = 0
        self.reconnects = 0

    # -- ciclo de vida ----------------------------------------------------

    def start(self) -> RtspCapture:
        if self._thread is not None:
            return self
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, name=f"capture:{self.name}", daemon=True
        )
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        self._new_frame.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    def __enter__(self) -> RtspCapture:
        return self.start()

    def __exit__(self, *_exc: object) -> None:
        self.stop()

    # -- consumo ----------------------------------------------------------

    def frames(self, timeout: float = 10.0) -> Iterator[Frame]:
        """Itera sobre os quadros mais recentes; nunca repete o mesmo indice."""
        last_index = -1
        while not self._stop.is_set():
            if not self._new_frame.wait(timeout=timeout):
                logger.warning("%s: sem quadro ha %.0fs", self.name, timeout)
                continue
            self._new_frame.clear()
            with self._lock:
                frame = self._latest
            if frame is not None and frame.index != last_index:
                last_index = frame.index
                yield frame

    def snapshot(self) -> Frame | None:
        with self._lock:
            return self._latest

    def ring(self) -> list[Frame]:
        """Copia do buffer circular -- serve de pre-roll para a evidencia."""
        with self._lock:
            return list(self._ring)

    # -- interno ----------------------------------------------------------

    def _loop(self) -> None:
        while not self._stop.is_set():
            capture = self._open()
            if capture is None:
                self._sleep(self.reconnect_delay)
                continue
            try:
                self._pump(capture)
            finally:
                capture.release()
            if not self._stop.is_set():
                self.reconnects += 1
                logger.warning(
                    "%s: fluxo caiu, reconectando em %.0fs",
                    self.name,
                    self.reconnect_delay,
                )
                self._sleep(self.reconnect_delay)

    def _open(self) -> Any:
        import cv2

        capture = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        try:
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:  # nem todo backend aceita
            pass
        if not capture.isOpened():
            logger.error("%s: falha ao abrir %s", self.name, _redact(self.url))
            capture.release()
            return None
        logger.info("%s: fluxo aberto", self.name)
        return capture

    def _pump(self, capture: Any) -> None:
        next_retrieve = time.monotonic()
        last_ok = time.monotonic()
        while not self._stop.is_set():
            if not capture.grab():
                if time.monotonic() - last_ok > self.open_timeout_s:
                    return
                time.sleep(0.01)
                continue
            grabbed_at = time.monotonic()
            last_ok = grabbed_at
            if grabbed_at < next_retrieve:
                continue  # descarta sem decodificar: mantem a latencia baixa
            ok, image = capture.retrieve()
            if not ok or image is None:
                continue
            next_retrieve = grabbed_at + self._interval
            self._publish(Frame(self._index, grabbed_at, time.time(), image))
            self._index += 1

    def _publish(self, frame: Frame) -> None:
        with self._lock:
            self._latest = frame
            self._ring.append(frame)
        self._new_frame.set()

    def _sleep(self, seconds: float) -> None:
        self._stop.wait(timeout=seconds)


def _redact(url: str) -> str:
    """Esconde a senha da URL RTSP antes de qualquer log."""
    if "@" not in url or "//" not in url:
        return url
    scheme, _, rest = url.partition("//")
    credentials, _, host = rest.rpartition("@")
    if not credentials:
        return url
    user, _, _password = credentials.partition(":")
    return f"{scheme}//{user}:***@{host}"

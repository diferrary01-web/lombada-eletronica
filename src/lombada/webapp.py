"""Servidor local de cadastro e teste de camera.

So biblioteca padrao: `http.server` basta para uma tela de configuracao que
roda na propria maquina, e evita arrastar um framework web para um projeto
cujo nucleo depende apenas de numpy e PyYAML.

**Escuta em 127.0.0.1 de proposito.** Esta tela mostra o resultado de sondagem
e grava URLs RTSP com senha; expo-la na rede entregaria as credenciais das
cameras a quem alcancasse a porta. Para abrir em outra interface e preciso
dizer `--host` explicitamente, e o servidor avisa no console.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .probe import probe_rtsp, samples_in_base
from .registry import CameraDraft, CameraRegistry, RegistryError
from .storage import PassageStore

logger = logging.getLogger(__name__)

PAGE = Path(__file__).parent / "web" / "index.html"
MAX_BODY = 1 << 20


def make_handler(registry: CameraRegistry, db_path: Path) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "lombada"
        protocol_version = "HTTP/1.1"

        # -- roteamento ------------------------------------------------

        def do_GET(self) -> None:  # noqa: N802
            path = urlparse(self.path).path.rstrip("/") or "/"
            try:
                if path == "/":
                    self._send_page()
                elif path == "/api/cameras":
                    self._json([c.to_public() for c in registry.cameras()])
                elif path == "/api/metrics":
                    self._json(self._metrics())
                elif path.startswith("/api/cameras/") and path.endswith("/snapshot"):
                    self._snapshot(path.split("/")[3])
                else:
                    self._json({"erro": "rota desconhecida"}, status=404)
            except Exception as exc:
                self._fail(exc)

        def do_POST(self) -> None:  # noqa: N802
            path = urlparse(self.path).path.rstrip("/") or "/"
            try:
                payload = self._body()
                if path == "/api/probe":
                    self._probe(payload)
                elif path == "/api/cameras":
                    draft = CameraDraft.from_payload(payload)
                    existente = registry.get(draft.id)
                    if existente and existente.image_points and not draft.image_points:
                        draft.image_points = existente.image_points
                    self._json(registry.save(draft).to_public(), status=201)
                elif path.startswith("/api/cameras/") and path.endswith("/calibrate"):
                    camera_id = path.split("/")[3]
                    saved = registry.calibrate(camera_id, payload.get("image_points"))
                    self._json(saved.to_public())
                else:
                    self._json({"erro": "rota desconhecida"}, status=404)
            except RegistryError as exc:
                self._json({"erro": str(exc)}, status=400)
            except Exception as exc:
                self._fail(exc)

        def do_DELETE(self) -> None:  # noqa: N802
            path = urlparse(self.path).path.rstrip("/") or "/"
            try:
                if path.startswith("/api/cameras/"):
                    removed = registry.delete(path.split("/")[3])
                    self._json({"removida": removed}, status=200 if removed else 404)
                else:
                    self._json({"erro": "rota desconhecida"}, status=404)
            except Exception as exc:
                self._fail(exc)

        # -- acoes -----------------------------------------------------

        def _probe(self, payload: dict[str, Any]) -> None:
            url = str(payload.get("rtsp_url", "")).strip()
            if not url:
                self._json({"erro": "informe a URL RTSP"}, status=400)
                return

            frames = int(payload.get("frames", 20))
            result = probe_rtsp(url, frames=max(2, min(frames, 120)))
            data = result.to_dict()

            # A conta que decide se a configuracao mede o que precisa medir.
            if result.ok:
                fps = result.fps_medido or result.fps_declarado
                distancia = float(payload.get("distance_m", 8.0) or 8.0)
                data["amostras"] = {
                    str(v): round(samples_in_base(fps, distancia, v), 1)
                    for v in (30, 40, 60, 80)
                }
            self._json(data)

        def _snapshot(self, camera_id: str) -> None:
            draft = registry.get(camera_id)
            if draft is None:
                self._json({"erro": "camera desconhecida"}, status=404)
                return

            import cv2

            source = draft.substream_url or draft.rtsp_url
            capture = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
            try:
                ok, frame = (False, None) if not capture.isOpened() else capture.read()
            finally:
                capture.release()

            if not ok or frame is None:
                self._json({"erro": "nao consegui ler um quadro"}, status=502)
                return

            ok, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 88])
            if not ok:
                self._json({"erro": "falha ao codificar o quadro"}, status=500)
                return

            payload = buffer.tobytes()
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)

        def _metrics(self) -> dict[str, Any]:
            store = PassageStore(db_path)
            since = datetime.now() - timedelta(days=7)
            recentes = store.recent(limit=25, since=since)
            return {
                "por_camera": store.stats(since=since),
                "passagens": [
                    {
                        "camera_id": row["camera_id"],
                        "captured_at": row["captured_at"],
                        "speed_kmh": round(row["speed_kmh"], 1),
                        "considered_kmh": round(row["considered_kmh"], 1),
                        "limit_kmh": row["limit_kmh"],
                        "is_violation": row["is_violation"],
                        "placa": row["plate"].text if row["plate"] else None,
                        "qualidade": row["quality"],
                    }
                    for row in recentes
                ],
            }

        # -- utilitarios ------------------------------------------------

        def _send_page(self) -> None:
            try:
                body = PAGE.read_bytes()
            except OSError:
                self._json({"erro": "pagina nao encontrada"}, status=500)
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _body(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length") or 0)
            if length <= 0:
                return {}
            if length > MAX_BODY:
                raise RegistryError("corpo da requisicao grande demais")
            raw = self.rfile.read(length)
            try:
                parsed = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise RegistryError(f"JSON invalido: {exc}") from exc
            return parsed if isinstance(parsed, dict) else {}

        def _json(self, data: Any, status: int = 200) -> None:
            body = json.dumps(data, ensure_ascii=False, default=str).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _fail(self, exc: Exception) -> None:
            logger.exception("erro em %s", self.path)
            self._json({"erro": f"{type(exc).__name__}: {exc}"}, status=500)

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            logger.debug("%s - %s", self.address_string(), format % args)

    return Handler


def create_server(
    *,
    host: str = "127.0.0.1",
    port: int = 8000,
    config_path: Path,
    db_path: Path,
) -> ThreadingHTTPServer:
    registry = CameraRegistry(path=config_path)
    registry.load()  # cria o arquivo com as secoes padrao se ainda nao existir
    return ThreadingHTTPServer((host, port), make_handler(registry, db_path))

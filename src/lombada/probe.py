"""Sondagem de um fluxo RTSP: o que a camera REALMENTE entrega.

Isto existe porque a camera mente com frequencia, e sempre em silencio:

- Um `subtype=1` de Dahua pode devolver o fluxo PRINCIPAL em 5 MP. O gatilho
  entao decodifica 5 MP achando que decodifica 640x360, come a GPU inteira e
  nada nos logs diz que isso esta acontecendo.
- O FPS declarado em `CAP_PROP_FPS` costuma ser o configurado, nao o entregue.
  Camera com rede ruim anuncia 25 e entrega 6.
- O tempo ate o primeiro quadro pode passar de dez segundos em H.265 com GOP
  longo, o que parece "camera off-line" para quem so espera dois segundos.

Por isso a sondagem MEDE em vez de perguntar: conta quadros com relogio na
mao e compara com o que a camera declara.

As constantes de propriedade estao em numero, e nao via `cv2.CAP_PROP_*`, de
proposito: assim toda a logica desta funcao e testavel sem OpenCV instalado.
Os valores fazem parte da ABI publica do OpenCV e nao mudam.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

CAP_PROP_FRAME_WIDTH = 3
CAP_PROP_FRAME_HEIGHT = 4
CAP_PROP_FPS = 5
CAP_PROP_FOURCC = 6

# Acima disso um fluxo "de gatilho" esta grande demais para o papel dele.
SUBSTREAM_MAX_PIXELS = 1_000_000


@dataclass
class ProbeResult:
    """O que a sondagem apurou sobre um fluxo."""

    ok: bool
    url_mascarada: str
    erro: str | None = None
    width: int = 0
    height: int = 0
    fps_declarado: float = 0.0
    fps_medido: float = 0.0
    primeiro_quadro_s: float = 0.0
    codec: str = ""
    quadros: int = 0
    avisos: list[str] = field(default_factory=list)

    @property
    def megapixels(self) -> float:
        return self.width * self.height / 1_000_000

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "url": self.url_mascarada,
            "erro": self.erro,
            "width": self.width,
            "height": self.height,
            "megapixels": round(self.megapixels, 2),
            "fps_declarado": round(self.fps_declarado, 1),
            "fps_medido": round(self.fps_medido, 1),
            "primeiro_quadro_s": round(self.primeiro_quadro_s, 2),
            "codec": self.codec,
            "quadros": self.quadros,
            "avisos": self.avisos,
        }


def probe_rtsp(
    url: str,
    *,
    frames: int = 20,
    capture_factory: Callable[[str], Any] | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> ProbeResult:
    """Abre o fluxo, le alguns quadros e mede o que chega de fato."""
    masked = mask_url(url)
    factory = capture_factory or _open_capture

    started = clock()
    try:
        capture = factory(url)
    except Exception as exc:  # driver ausente, URL malformada
        return ProbeResult(ok=False, url_mascarada=masked, erro=str(exc))

    try:
        if not capture.isOpened():
            return ProbeResult(
                ok=False,
                url_mascarada=masked,
                erro="nao foi possivel abrir o fluxo (URL, credencial ou rede)",
            )

        ok, frame = capture.read()
        first_frame_at = clock()
        if not ok or frame is None:
            return ProbeResult(
                ok=False,
                url_mascarada=masked,
                erro="fluxo abriu mas nao entregou quadro",
                primeiro_quadro_s=first_frame_at - started,
            )

        height, width = frame.shape[0], frame.shape[1]
        read = 1
        timestamps = [first_frame_at]
        for _ in range(max(0, frames - 1)):
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            read += 1
            timestamps.append(clock())

        measured = _fps(timestamps)
        declared = _float(capture, CAP_PROP_FPS)
        result = ProbeResult(
            ok=True,
            url_mascarada=masked,
            width=width,
            height=height,
            fps_declarado=declared,
            fps_medido=measured,
            primeiro_quadro_s=first_frame_at - started,
            codec=_fourcc(capture),
            quadros=read,
        )
        result.avisos = _warnings(result)
        return result
    finally:
        _release(capture)


def _warnings(result: ProbeResult) -> list[str]:
    avisos: list[str] = []

    if result.width * result.height > SUBSTREAM_MAX_PIXELS:
        avisos.append(
            f"fluxo de {result.megapixels:.1f} MP ({result.width}x{result.height}). "
            "Se voce cadastrou isto como SUBSTREAM, confira: algumas Dahua "
            "devolvem o fluxo principal quando se pede o secundario, e o "
            "gatilho passa a decodificar imagem cheia sem avisar."
        )

    if result.fps_declarado > 0 and result.fps_medido > 0:
        razao = result.fps_medido / result.fps_declarado
        if razao < 0.6:
            avisos.append(
                f"a camera declara {result.fps_declarado:.0f} fps mas entregou "
                f"{result.fps_medido:.1f}. Rede, CPU ou a propria camera nao "
                "estao dando conta -- a medida de velocidade fica com menos "
                "amostras do que a configuracao sugere."
            )

    if result.primeiro_quadro_s > 5.0:
        avisos.append(
            f"{result.primeiro_quadro_s:.1f}s ate o primeiro quadro. GOP longo "
            "ou rede lenta; nao confunda com camera off-line."
        )

    if 0 < result.fps_medido < 5:
        avisos.append(
            f"{result.fps_medido:.1f} fps e pouco para medir velocidade: um "
            "veiculo a 40 km/h cruza a base de 8 m em 0,7 s."
        )

    if result.quadros < 5:
        avisos.append(
            f"so {result.quadros} quadros lidos; o fluxo pode estar instavel."
        )

    return avisos


def samples_in_base(fps: float, distance_m: float, speed_kmh: float) -> float:
    """Quantas amostras cabem na base a uma dada velocidade.

    E a conta que decide se a configuracao consegue medir a velocidade que
    interessa: abaixo de ~4 amostras o ajuste nao se sustenta e a passagem e
    recusada -- o sistema silenciosamente deixa de ver os carros rapidos.
    """
    if speed_kmh <= 0 or fps <= 0:
        return 0.0
    return distance_m / (speed_kmh / 3.6) * fps


def mask_url(url: str) -> str:
    """Esconde a senha antes de qualquer log, tela ou resposta HTTP."""
    if "@" not in url or "//" not in url:
        return url
    scheme, _, rest = url.partition("//")
    credentials, _, host = rest.rpartition("@")
    if not credentials:
        return url
    user, _, _password = credentials.partition(":")
    return f"{scheme}//{user}:***@{host}"


def _open_capture(url: str) -> Any:
    import cv2

    return cv2.VideoCapture(url, cv2.CAP_FFMPEG)


def _fps(timestamps: list[float]) -> float:
    if len(timestamps) < 2:
        return 0.0
    elapsed = timestamps[-1] - timestamps[0]
    return (len(timestamps) - 1) / elapsed if elapsed > 0 else 0.0


def _float(capture: Any, prop: int) -> float:
    try:
        value = float(capture.get(prop))
    except Exception:
        return 0.0
    return value if value > 0 and value < 1000 else 0.0


def _fourcc(capture: Any) -> str:
    try:
        code = int(capture.get(CAP_PROP_FOURCC))
    except Exception:
        return ""
    if code <= 0:
        return ""
    chars = [chr((code >> (8 * i)) & 0xFF) for i in range(4)]
    return "".join(c for c in chars if c.isprintable()).strip()


def _release(capture: Any) -> None:
    try:
        capture.release()
    except Exception:
        pass

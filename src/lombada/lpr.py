"""Leitura de placa: deteccao, OCR e consolidacao entre quadros.

O ganho grande aqui nao vem do modelo: vem de ler VARIOS quadros da mesma
passagem e votar caractere a caractere. Uma unica leitura erra um digito com
frequencia; a votacao posicional ponderada pela confianca corrige quase todos
esses casos, porque o erro raramente cai na mesma posicao duas vezes.

A validacao de formato (placa antiga `LLLNNNN` e Mercosul `LLLNLNN`) entra
depois da votacao, so para desambiguar confusoes classicas de OCR -- `O`/`0`,
`I`/`1`, `S`/`5` -- em posicoes cujo tipo o formato ja determina. A posicao 5
(1-indexada) e a unica que os dois formatos disputam, e por isso nunca e
coagida: e ela que decide qual formato e.
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from collections.abc import Sequence
from typing import Any, Protocol

from .config import LprConfig
from .models import BBox, PlateRead

logger = logging.getLogger(__name__)

PLATE_LENGTH = 7
OLD_FORMAT = re.compile(r"^[A-Z]{3}[0-9]{4}$")
MERCOSUL_FORMAT = re.compile(r"^[A-Z]{3}[0-9][A-Z][0-9]{2}$")

# Posicoes (0-indexadas) cujo tipo os DOIS formatos concordam.
_LETTER_POSITIONS = (0, 1, 2)
_DIGIT_POSITIONS = (3, 5, 6)

_TO_LETTER = {
    "0": "O", "1": "I", "2": "Z", "4": "A",
    "5": "S", "6": "G", "7": "T", "8": "B",
}
_TO_DIGIT = {
    "O": "0", "Q": "0", "D": "0", "I": "1", "L": "1", "Z": "2",
    "A": "4", "S": "5", "G": "6", "T": "7", "B": "8",
}


class PlateReader(Protocol):
    """Contrato de um leitor de placa."""

    def read(self, image: Any) -> PlateRead | None:
        """Le a placa de UM recorte de veiculo (BGR). `None` se nao achou."""
        ...


class NullPlateReader:
    """Nao le nada. Permite operar a lombada so como medidor de velocidade."""

    def read(self, image: Any) -> PlateRead | None:  # noqa: ARG002
        return None


class FastPlateOcrReader:
    """`open_image_models` para localizar a placa, `fast_plate_ocr` para ler.

    Os dois rodam em ONNX Runtime. Se o pacote `onnxruntime-gpu` estiver
    instalado sem as bibliotecas do TensorRT, a lib imprime um bloco de erro
    do provedor TRT a cada modelo carregado e cai sozinha para CUDA -- passar
    os providers explicitamente evita o ruido e torna a escolha auditavel.
    """

    def __init__(self, config: LprConfig) -> None:
        self.config = config
        self._detector: Any = None
        self._recognizer: Any = None

    @property
    def _providers(self) -> list[str]:
        if self.config.device.startswith("cuda"):
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        return ["CPUExecutionProvider"]

    def _load(self) -> tuple[Any, Any]:
        if self._detector is None:
            from open_image_models import LicensePlateDetector

            self._detector = LicensePlateDetector(
                detection_model=self.config.detector_model
            )
        if self._recognizer is None:
            from fast_plate_ocr import LicensePlateRecognizer

            self._recognizer = LicensePlateRecognizer(
                hub_ocr_model=self.config.recognizer_model,
                providers=self._providers,
            )
        return self._detector, self._recognizer

    def read(self, image: Any) -> PlateRead | None:
        detector, recognizer = self._load()
        found = detector.predict(image)
        if not found:
            return None

        best = max(found, key=lambda d: float(getattr(d, "confidence", 0.0)))
        box = best.bounding_box
        x1, y1, x2, y2 = int(box.x1), int(box.y1), int(box.x2), int(box.y2)
        crop = image[max(0, y1) : max(0, y2), max(0, x1) : max(0, x2)]
        if crop.size == 0:
            return None

        texts, probabilities = recognizer.run(crop, return_confidence=True)
        if not texts:
            return None
        text = str(texts[0]).strip().upper()
        confidence = float(min(probabilities[0])) if len(probabilities) else 0.0
        if not text or confidence < self.config.min_confidence:
            return None
        return PlateRead(text=text, confidence=confidence, bbox=(x1, y1, x2, y2))


_BACKENDS: dict[str, Any] = {
    "fast_plate_ocr": FastPlateOcrReader,
    "stub": lambda _config: NullPlateReader(),
}


def build_reader(config: LprConfig) -> PlateReader:
    if not config.enabled:
        return NullPlateReader()
    try:
        factory = _BACKENDS[config.backend]
    except KeyError:
        raise ValueError(
            f"backend de LPR desconhecido: {config.backend!r} "
            f"(disponiveis: {sorted(_BACKENDS)})"
        ) from None
    return factory(config)


# -- consolidacao entre quadros (puro, testavel sem modelo) ----------------


def vote_plates(
    reads: Sequence[PlateRead], *, min_votes: int = 1
) -> PlateRead | None:
    """Consolida varias leituras da mesma passagem em uma so.

    Vota caractere a caractere, ponderando cada voto pela confianca da leitura
    de onde veio. Leituras com comprimento diferente do padrao entram na
    votacao do texto inteiro, mas nao na posicional -- alinhar strings de
    tamanhos diferentes so espalharia o erro.
    """
    usable = [r for r in reads if r.text]
    if len(usable) < min_votes or not usable:
        return None

    aligned = [r for r in usable if len(r.text) == PLATE_LENGTH]
    if not aligned:
        best = max(usable, key=lambda r: r.confidence)
        return PlateRead(text=best.text, confidence=best.confidence, bbox=best.bbox)

    chars: list[str] = []
    scores: list[float] = []
    for position in range(PLATE_LENGTH):
        tally: dict[str, float] = defaultdict(float)
        for read in aligned:
            tally[read.text[position]] += max(read.confidence, 1e-6)
        char, weight = max(tally.items(), key=lambda kv: kv[1])
        total = sum(tally.values())
        chars.append(char)
        scores.append(weight / total if total else 0.0)

    text = coerce_format("".join(chars))
    # A confianca do conjunto e a do caractere MAIS FRACO: uma placa so serve
    # se todos os caracteres servem.
    confidence = min(scores) if scores else 0.0
    bbox = max(aligned, key=lambda r: r.confidence).bbox
    return PlateRead(text=text, confidence=confidence, bbox=bbox)


def normalize_plate(text: str) -> str:
    """Maiusculas, sem separadores nem caracteres fora de A-Z0-9."""
    return re.sub(r"[^A-Z0-9]", "", text.upper())


def coerce_format(text: str) -> str:
    """Corrige confusoes de OCR nas posicoes cujo tipo o formato determina."""
    plate = normalize_plate(text)
    if len(plate) != PLATE_LENGTH:
        return plate
    chars = list(plate)
    for position in _LETTER_POSITIONS:
        chars[position] = _TO_LETTER.get(chars[position], chars[position])
    for position in _DIGIT_POSITIONS:
        chars[position] = _TO_DIGIT.get(chars[position], chars[position])
    return "".join(chars)


def plate_format(text: str) -> str | None:
    """`"mercosul"`, `"antiga"` ou `None` se nao casa com nenhum formato."""
    plate = normalize_plate(text)
    if MERCOSUL_FORMAT.match(plate):
        return "mercosul"
    if OLD_FORMAT.match(plate):
        return "antiga"
    return None


def is_plausible(text: str) -> bool:
    return plate_format(text) is not None


def crop_bbox(image: Any, bbox: BBox, *, margin: float = 0.05) -> Any:
    """Recorta a bbox com uma folga relativa, sem sair dos limites do quadro."""
    height, width = image.shape[:2]
    x1, y1, x2, y2 = bbox
    pad_x = (x2 - x1) * margin
    pad_y = (y2 - y1) * margin
    x1 = max(0, int(x1 - pad_x))
    y1 = max(0, int(y1 - pad_y))
    x2 = min(width, int(x2 + pad_x))
    y2 = min(height, int(y2 + pad_y))
    if x2 <= x1 or y2 <= y1:
        return image[0:0, 0:0]
    return image[y1:y2, x1:x2]

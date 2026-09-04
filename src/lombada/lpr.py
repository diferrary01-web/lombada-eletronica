"""Leitura de placa por ENSEMBLE de motores, com votacao caractere a caractere.

O ganho grande aqui nao vem de escolher "o melhor modelo": vem de ler a mesma
passagem com motores que erram DIFERENTE e votar. Um reconhecedor CTC e um
seq2seq com atencao falham em coisas distintas -- borrao, inclinacao, sujeira,
caractere colado -- e a interseccao dos dois erros e bem menor que cada um.

A votacao acontece em duas dimensoes ao mesmo tempo e no mesmo lugar:

- entre QUADROS: a mesma placa aparece varias vezes na passagem;
- entre MOTORES: cada quadro rende uma leitura por motor.

Por isso `PlateRead` carrega `source` e `weight`: o voto precisa distinguir
"dois motores independentes concordaram" de "o mesmo motor repetiu o mesmo erro
seis vezes seguidas".

Papeis:

- `PlateReader.read_all(recorte_do_veiculo)` -> varias leituras. E o que o
  pipeline chama.
- `PlateRecognizer.recognize(recorte_da_placa)` -> uma leitura. Motor que so le
  uma linha de texto e nao sabe localizar nada (TrOCR e o caso).

O `EnsembleReader` costura os dois: o motor primario localiza e le, a melhor
caixa dele vira o recorte da placa, e esse recorte alimenta os secundarios.
Assim nenhum detector extra entra so para servir o TrOCR.
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

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

# Uma placa brasileira mede 400x130 mm: razao ~3:1 de frente, menos de lado.
# A faixa aceita e larga de proposito -- serve so para descartar adesivo,
# emblema do modelo e faixa de concessionaria, nao para validar placa.
_MIN_ASPECT = 1.5
_MAX_ASPECT = 7.0


class PlateReader(Protocol):
    """Le um recorte de VEICULO e devolve todas as leituras que conseguiu."""

    name: str

    def read_all(self, image: Any) -> list[PlateRead]:
        ...


@runtime_checkable
class PlateRecognizer(Protocol):
    """Le um recorte de PLACA ja localizado. Nao localiza nada sozinho."""

    name: str

    def recognize(self, crop: Any) -> PlateRead | None:
        ...


class NullPlateReader:
    """Nao le nada. Permite operar a lombada so como medidor de velocidade."""

    name = "null"

    def read_all(self, image: Any) -> list[PlateRead]:  # noqa: ARG002
        return []

    def recognize(self, crop: Any) -> PlateRead | None:  # noqa: ARG002
        return None


class RapidOcrEngine:
    """PP-OCR sobre ONNX Runtime, via `rapidocr`.

    Localiza E le, entao serve de motor primario do ensemble. O dicionario de
    caracteres e publico (ao contrario de varios pacotes de LPR que embutem o
    charset no binario), o que torna o resultado auditavel.

    CUIDADO NA INSTALACAO: o `rapidocr` declara `opencv-python-headless` SEM
    fixar versao. Foi por essa porta que o OpenCV 5.0 entrou num parque de
    cameras e derrubou a decodificacao RTSP inteira. O extra `[lpr]` do
    pyproject repete o pin de proposito -- nao instale este pacote solto.
    """

    name = "rapidocr"

    def __init__(self, config: LprConfig, engine: Any = None) -> None:
        self.config = config
        self._engine = engine

    def _load(self) -> Any:
        if self._engine is None:
            from rapidocr import RapidOCR

            logger.info("carregando RapidOCR (PP-OCR / ONNX Runtime)")
            self._engine = RapidOCR()
        return self._engine

    def read_all(self, image: Any) -> list[PlateRead]:
        result = self._load()(image, use_det=True, use_cls=True, use_rec=True)
        texts = getattr(result, "txts", None)
        if texts is None or len(texts) == 0:
            return []

        # `boxes` volta como ndarray (N,4,2). `x or []` num ndarray levanta
        # "truth value of an array is ambiguous" -- por isso o teste e contra
        # None, e nao um `or`.
        raw_scores = getattr(result, "scores", None)
        raw_boxes = getattr(result, "boxes", None)
        scores = [] if raw_scores is None else list(raw_scores)
        boxes = [] if raw_boxes is None else list(raw_boxes)

        reads: list[PlateRead] = []
        for index, raw_text in enumerate(texts):
            text = normalize_plate(str(raw_text))
            if not text:
                continue
            bbox = _polygon_to_bbox(boxes[index]) if index < len(boxes) else None
            if bbox is not None and not _plate_shaped(bbox):
                continue
            confidence = float(scores[index]) if index < len(scores) else 0.0
            if confidence < self.config.min_confidence:
                continue
            reads.append(
                PlateRead(
                    text=text,
                    confidence=confidence,
                    bbox=bbox,
                    source=self.name,
                    weight=self.config.weight_of(self.name),
                )
            )
        return reads

    def recognize(self, crop: Any) -> PlateRead | None:
        reads = self.read_all(crop)
        return max(reads, key=lambda r: r.confidence) if reads else None


class TrOcrEngine:
    """TrOCR (`microsoft/trocr-*-printed`): encoder de imagem + decoder seq2seq.

    Entra no ensemble justamente por NAO ser um CTC: ele decodifica com
    atencao, entao erra em situacoes diferentes das do PP-OCR. Nao localiza
    nada -- recebe o recorte da placa que o motor primario achou.

    A confianca devolvida e a do caractere MAIS FRACO da sequencia, nao a
    media: uma placa so serve se todos os caracteres servem, e a media esconde
    exatamente o caractere que vai estar errado.
    """

    name = "trocr"

    def __init__(self, config: LprConfig, model: Any = None, processor: Any = None):
        self.config = config
        self._model = model
        self._processor = processor

    def _load(self) -> tuple[Any, Any]:
        if self._model is None or self._processor is None:
            import torch
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel

            logger.info("carregando TrOCR %s", self.config.trocr_model)
            self._processor = TrOCRProcessor.from_pretrained(self.config.trocr_model)
            # Anotado como Any de proposito: em transformers 5.x o `.to()` e
            # decorado de um jeito que o mypy resolve como `_Wrapped` e passa a
            # exigir um `PreTrainedModel` onde vai a string do device. E falso
            # positivo, e so aparece onde o transformers ESTA instalado -- a CI
            # nao instala os extras pesados, entao isto passaria batido la.
            model: Any = VisionEncoderDecoderModel.from_pretrained(
                self.config.trocr_model
            )
            device = self.config.device if torch.cuda.is_available() else "cpu"
            self._model = model.to(device).eval()
        return self._model, self._processor

    def recognize(self, crop: Any) -> PlateRead | None:
        if getattr(crop, "size", 0) == 0:
            return None
        import torch

        model, processor = self._load()
        pixel_values = processor(images=_to_rgb(crop), return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(model.device)

        with torch.no_grad():
            output = model.generate(
                pixel_values,
                max_new_tokens=16,
                output_scores=True,
                return_dict_in_generate=True,
            )

        text = normalize_plate(
            processor.batch_decode(output.sequences, skip_special_tokens=True)[0]
        )
        if not text:
            return None

        probabilities = [
            float(torch.softmax(step[0], dim=-1).max()) for step in output.scores
        ]
        confidence = min(probabilities) if probabilities else 0.0
        if confidence < self.config.min_confidence:
            return None
        return PlateRead(
            text=text,
            confidence=confidence,
            source=self.name,
            weight=self.config.weight_of(self.name),
        )

    def read_all(self, image: Any) -> list[PlateRead]:
        """Sem localizador, le a imagem inteira como uma linha de texto.

        So faz sentido quando o recorte JA e a placa. Como motor solto num
        recorte de veiculo, o resultado e ruido -- use dentro do ensemble.
        """
        read = self.recognize(image)
        return [read] if read else []


class FastPlateOcrEngine:
    """`open_image_models` para localizar + `fast_plate_ocr` para ler.

    Fica disponivel como motor opcional, mas NAO entra no ensemble padrao.
    """

    name = "fast_plate_ocr"

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

    def recognize(self, crop: Any) -> PlateRead | None:
        if getattr(crop, "size", 0) == 0:
            return None
        _detector, recognizer = self._load()
        texts, probabilities = recognizer.run(crop, return_confidence=True)
        if not texts:
            return None
        text = normalize_plate(str(texts[0]))
        confidence = float(min(probabilities[0])) if len(probabilities) else 0.0
        if not text or confidence < self.config.min_confidence:
            return None
        return PlateRead(
            text=text,
            confidence=confidence,
            source=self.name,
            weight=self.config.weight_of(self.name),
        )

    def read_all(self, image: Any) -> list[PlateRead]:
        detector, _recognizer = self._load()
        found = detector.predict(image)
        reads: list[PlateRead] = []
        for detection in found:
            box = detection.bounding_box
            bbox = (float(box.x1), float(box.y1), float(box.x2), float(box.y2))
            crop = crop_bbox(image, bbox, margin=0.0)
            read = self.recognize(crop)
            if read is not None:
                reads.append(
                    PlateRead(
                        text=read.text,
                        confidence=read.confidence,
                        bbox=bbox,
                        source=read.source,
                        weight=read.weight,
                    )
                )
        return reads


class EnsembleReader:
    """Motor primario localiza e le; os secundarios releem o mesmo recorte."""

    name = "ensemble"

    def __init__(
        self, primary: PlateReader, secondaries: Sequence[PlateRecognizer] = ()
    ) -> None:
        self.primary = primary
        self.secondaries = list(secondaries)

    @property
    def engine_names(self) -> list[str]:
        return [self.primary.name] + [s.name for s in self.secondaries]

    def read_all(self, image: Any) -> list[PlateRead]:
        reads = list(_safe(self.primary.read_all, image, self.primary.name) or [])
        if not reads or not self.secondaries:
            return reads

        best = max(reads, key=lambda r: r.confidence)
        crop = crop_bbox(image, best.bbox, margin=0.08) if best.bbox else image
        if getattr(crop, "size", 1) == 0:
            return reads

        for engine in self.secondaries:
            read = _safe(engine.recognize, crop, engine.name)
            if read is not None:
                reads.append(read)
        return reads


_ENGINES: dict[str, Any] = {
    "rapidocr": RapidOcrEngine,
    "trocr": TrOcrEngine,
    "fast_plate_ocr": FastPlateOcrEngine,
    "stub": lambda _config: NullPlateReader(),
}


def build_engine(name: str, config: LprConfig) -> Any:
    try:
        factory = _ENGINES[name]
    except KeyError:
        raise ValueError(
            f"motor de LPR desconhecido: {name!r} (disponiveis: {sorted(_ENGINES)})"
        ) from None
    return factory(config)


def build_reader(config: LprConfig) -> PlateReader:
    """Monta o leitor a partir da configuracao.

    Com um motor so, devolve o motor direto -- nao ha o que costurar. Com dois
    ou mais, o primeiro e o primario (precisa saber localizar) e os demais
    releem o recorte que ele achou.
    """
    if not config.enabled:
        return NullPlateReader()

    names = list(config.engines) or ["rapidocr"]
    engines = [build_engine(name, config) for name in names]
    if len(engines) == 1:
        return engines[0]

    secondaries = [e for e in engines[1:] if hasattr(e, "recognize")]
    return EnsembleReader(engines[0], secondaries)


# -- consolidacao (puro, testavel sem nenhum modelo) -----------------------


def vote_plates(reads: Sequence[PlateRead], *, min_votes: int = 1) -> PlateRead | None:
    """Consolida as leituras de uma passagem em uma so.

    Vota caractere a caractere, com cada voto pesando `confianca x peso do
    motor`. Leituras de comprimento diferente do padrao entram na escolha do
    texto inteiro, mas nao na votacao posicional -- alinhar strings de tamanhos
    diferentes so espalharia o erro por todas as posicoes.

    A massa de voto e NORMALIZADA POR MOTOR: cada motor contribui com o mesmo
    total, independente de quantos quadros ele leu. Sem isso o motor que
    devolveu leitura em seis quadros afogaria o que devolveu em um, e o
    ensemble viraria uma eleicao decidida por quem falou mais alto -- que e
    exatamente o erro sistematico que o ensemble existe para evitar. Leituras
    sem `source` contam como um unico motor anonimo.
    """
    usable = [r for r in reads if r.text]
    if not usable or len(usable) < min_votes:
        return None

    aligned = [r for r in usable if len(normalize_plate(r.text)) == PLATE_LENGTH]
    if not aligned:
        best = max(usable, key=lambda r: r.vote_weight)
        return best

    per_source: dict[str, int] = defaultdict(int)
    for read in aligned:
        per_source[read.source] += 1

    chars: list[str] = []
    scores: list[float] = []
    for position in range(PLATE_LENGTH):
        tally: dict[str, float] = defaultdict(float)
        for read in aligned:
            share = read.vote_weight / per_source[read.source]
            tally[normalize_plate(read.text)[position]] += share
        char, weight = max(tally.items(), key=lambda kv: kv[1])
        total = sum(tally.values())
        chars.append(char)
        scores.append(weight / total if total else 0.0)

    text = coerce_format("".join(chars))
    # A confianca do conjunto e a do caractere MAIS FRACO: uma placa so serve
    # se todos os caracteres servem.
    confidence = min(scores) if scores else 0.0
    best = max(aligned, key=lambda r: r.vote_weight)
    return PlateRead(
        text=text,
        confidence=confidence,
        bbox=best.bbox,
        source="+".join(sorted({r.source for r in aligned if r.source})),
    )


def agreement(reads: Sequence[PlateRead]) -> float:
    """Fracao dos motores DISTINTOS que leram o texto majoritario.

    Serve de sinal de qualidade independente da confianca que cada modelo
    declara: modelo confiante e errado e comum, dois modelos independentes
    confiantes e errados no mesmo caractere e raro.
    """
    by_source: dict[str, str] = {}
    for read in reads:
        if read.text and read.source and read.source not in by_source:
            by_source[read.source] = coerce_format(read.text)
    if not by_source:
        return 0.0
    counts: dict[str, int] = defaultdict(int)
    for text in by_source.values():
        counts[text] += 1
    return max(counts.values()) / len(by_source)


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


def crop_bbox(image: Any, bbox: BBox | None, *, margin: float = 0.05) -> Any:
    """Recorta a bbox com uma folga relativa, sem sair dos limites do quadro."""
    if bbox is None:
        return image
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


def _polygon_to_bbox(polygon: Any) -> BBox | None:
    """Converte o quadrilatero do detector em caixa alinhada aos eixos."""
    try:
        points = [(float(p[0]), float(p[1])) for p in polygon]
    except (TypeError, IndexError, ValueError):
        return None
    if not points:
        return None
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return (min(xs), min(ys), max(xs), max(ys))


def _plate_shaped(bbox: BBox) -> bool:
    """Descarta caixa de texto que nao tem proporcao de placa."""
    x1, y1, x2, y2 = bbox
    width, height = x2 - x1, y2 - y1
    if width <= 0 or height <= 0:
        return False
    return _MIN_ASPECT <= width / height <= _MAX_ASPECT


def _to_rgb(image: Any) -> Any:
    """BGR do OpenCV -> RGB, que e o que os modelos do `transformers` esperam."""
    if getattr(image, "ndim", 0) == 3 and image.shape[2] == 3:
        return image[:, :, ::-1]
    return image


def _safe(call: Any, argument: Any, name: str) -> Any:
    """Um motor que explode nao pode derrubar os outros nem a passagem."""
    try:
        return call(argument)
    except Exception:
        logger.exception("motor de LPR %s falhou", name)
        return None

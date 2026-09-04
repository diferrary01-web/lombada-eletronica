"""Bancada de avaliacao do ensemble contra recortes reais com gabarito.

A pergunta que esta bancada responde nao e "qual a acuracia" -- e **o ensemble
se paga?**. Um ensemble pode ser pior que o melhor motor sozinho: se um motor
fraco tem confianca inflada, ele arrasta a votacao e estraga leituras que o
motor forte acertaria. Isso acontece, e nao aparece numa media global.

Por isso as duas metricas que decidem sao:

- `ensemble_ganhos`: acertou onde NENHUM motor sozinho acertou. E a
  justificativa da existencia do ensemble.
- `ensemble_regressoes`: errou onde ALGUM motor sozinho acertou. E o preco.

Se as regressoes superarem os ganhos, a resposta honesta e usar o melhor motor
sozinho -- ou ajustar `engine_weights` ate a conta virar.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .lpr import agreement, coerce_format, normalize_plate, vote_plates
from .models import PlateRead


@dataclass(frozen=True)
class Sample:
    """Um recorte com gabarito."""

    path: Path
    truth: str

    @property
    def expected(self) -> str:
        return normalize_plate(self.truth)


@dataclass
class EngineScore:
    """Desempenho de um motor isolado."""

    engine: str
    seen: int = 0  # imagens em que devolveu alguma leitura
    exact: int = 0
    cer_total: float = 0.0

    @property
    def exact_rate(self) -> float:
        return self.exact / self.seen if self.seen else 0.0

    @property
    def cer(self) -> float:
        return self.cer_total / self.seen if self.seen else 1.0


@dataclass
class BenchResult:
    total: int = 0
    ensemble_exact: int = 0
    ensemble_cer_total: float = 0.0
    sem_leitura: int = 0
    ensemble_ganhos: int = 0
    ensemble_regressoes: int = 0
    engines: dict[str, EngineScore] = field(default_factory=dict)
    agreement_total: float = 0.0
    erros: list[tuple[str, str, str]] = field(default_factory=list)

    @property
    def ensemble_exact_rate(self) -> float:
        return self.ensemble_exact / self.total if self.total else 0.0

    @property
    def ensemble_cer(self) -> float:
        return self.ensemble_cer_total / self.total if self.total else 1.0

    @property
    def mean_agreement(self) -> float:
        return self.agreement_total / self.total if self.total else 0.0

    @property
    def cobertura(self) -> float:
        """Fracao das imagens em que saiu alguma placa."""
        return 1.0 - (self.sem_leitura / self.total) if self.total else 0.0

    def veredito(self) -> str:
        if self.total == 0:
            return "sem amostras"
        if self.ensemble_ganhos > self.ensemble_regressoes:
            return (
                f"o ensemble se paga: +{self.ensemble_ganhos} ganhos contra "
                f"-{self.ensemble_regressoes} regressoes"
            )
        if self.ensemble_ganhos == self.ensemble_regressoes:
            return (
                f"empate ({self.ensemble_ganhos} x {self.ensemble_regressoes}): "
                "o ensemble nao esta pagando o custo de rodar dois motores"
            )
        melhor = self.melhor_motor()
        return (
            f"o ensemble PIORA: {self.ensemble_ganhos} ganhos contra "
            f"{self.ensemble_regressoes} regressoes"
            + (f" -- considere usar so `{melhor}`" if melhor else "")
        )

    def melhor_motor(self) -> str | None:
        if not self.engines:
            return None
        return max(self.engines.values(), key=lambda e: e.exact_rate).engine


def evaluate(
    reader: Any,
    samples: Sequence[Sample],
    load_image: Callable[[Path], Any],
) -> BenchResult:
    """Roda o leitor sobre as amostras e apura as metricas.

    `load_image` e injetado para que a bancada seja testavel sem OpenCV e sem
    arquivo em disco.
    """
    result = BenchResult()
    for sample in samples:
        result.total += 1
        expected = sample.expected

        image = load_image(sample.path)
        if image is None:
            result.sem_leitura += 1
            result.ensemble_cer_total += 1.0
            continue

        reads = list(reader.read_all(image))
        _score_engines(result, reads, expected)
        result.agreement_total += agreement(reads)

        voted = vote_plates(reads)
        predicted = coerce_format(voted.text) if voted else ""
        if not predicted:
            result.sem_leitura += 1
            result.ensemble_cer_total += 1.0
            continue

        acertou = predicted == expected
        result.ensemble_exact += int(acertou)
        result.ensemble_cer_total += character_error_rate(predicted, expected)
        if not acertou:
            result.erros.append((sample.path.name, expected, predicted))

        algum_motor_acertou = any(
            coerce_format(r.text) == expected for r in reads if r.text
        )
        if acertou and not algum_motor_acertou:
            result.ensemble_ganhos += 1
        elif not acertou and algum_motor_acertou:
            result.ensemble_regressoes += 1

    return result


def _score_engines(
    result: BenchResult, reads: Iterable[PlateRead], expected: str
) -> None:
    """Pontua cada motor uma vez por imagem, pela melhor leitura dele.

    Uma vez POR IMAGEM, e nao por leitura: senao o motor que devolve varias
    leituras por recorte apareceria com mais peso na estatistica do que o que
    devolve uma -- o mesmo vies que a votacao ja corrige.
    """
    melhor: dict[str, PlateRead] = {}
    for read in reads:
        if not read.text:
            continue
        atual = melhor.get(read.source)
        if atual is None or read.confidence > atual.confidence:
            melhor[read.source] = read

    for source, read in melhor.items():
        score = result.engines.setdefault(source, EngineScore(engine=source))
        predicted = coerce_format(read.text)
        score.seen += 1
        score.exact += int(predicted == expected)
        score.cer_total += character_error_rate(predicted, expected)


def character_error_rate(predicted: str, expected: str) -> float:
    """Distancia de edicao normalizada pelo tamanho do gabarito.

    Mais informativo que acerto exato numa bancada pequena: distingue "errou um
    caractere" de "leu outra placa", e as duas coisas tem consequencia bem
    diferente na operacao.
    """
    if not expected:
        return 0.0 if not predicted else 1.0
    return levenshtein(predicted, expected) / len(expected)


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    previous = list(range(len(b) + 1))
    for i, char_a in enumerate(a, start=1):
        current = [i]
        for j, char_b in enumerate(b, start=1):
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + (char_a != char_b),
                )
            )
        previous = current
    return previous[-1]


def load_samples(directory: Path, labels: Path | None = None) -> list[Sample]:
    """Monta as amostras de um diretorio.

    Sem `labels`, o gabarito e o NOME DO ARQUIVO (`ABC1D23.jpg`) -- e o jeito
    mais rapido de montar um corpus a mao. Com `labels`, le um CSV
    `arquivo,placa`, que e o que sai da exportacao automatica.
    """
    if labels is not None:
        return _from_csv(directory, labels)

    samples: list[Sample] = []
    for path in sorted(directory.iterdir()):
        if path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        truth = normalize_plate(path.stem.split("_")[0])
        if truth:
            samples.append(Sample(path=path, truth=truth))
    return samples


def _from_csv(directory: Path, labels: Path) -> list[Sample]:
    import csv

    samples: list[Sample] = []
    with open(labels, encoding="utf-8", newline="") as handle:
        for row in csv.reader(handle):
            if len(row) < 2 or row[0].strip().lower() in ("arquivo", "file"):
                continue
            truth = normalize_plate(row[1])
            if truth:
                samples.append(Sample(path=directory / row[0].strip(), truth=truth))
    return samples


def format_report(result: BenchResult, *, max_errors: int = 15) -> str:
    """Relatorio de texto da bancada."""
    if result.total == 0:
        return "nenhuma amostra encontrada"

    linhas = [
        f"amostras: {result.total}",
        f"cobertura (saiu alguma placa): {result.cobertura:6.1%}",
        f"concordancia media entre motores: {result.mean_agreement:6.1%}",
        "",
        f"{'motor':<20} {'imagens':>8} {'exato':>8} {'CER':>8}",
        "-" * 48,
    ]
    for score in sorted(
        result.engines.values(), key=lambda e: e.exact_rate, reverse=True
    ):
        linhas.append(
            f"{score.engine:<20} {score.seen:>8} "
            f"{score.exact_rate:>7.1%} {score.cer:>8.3f}"
        )
    linhas.append(
        f"{'ENSEMBLE':<20} {result.total:>8} "
        f"{result.ensemble_exact_rate:>7.1%} {result.ensemble_cer:>8.3f}"
    )
    linhas += [
        "",
        f"ganhos do ensemble (so ele acertou):      {result.ensemble_ganhos}",
        f"regressoes (motor sozinho acertaria):     {result.ensemble_regressoes}",
        f"-> {result.veredito()}",
    ]

    if result.erros:
        linhas += ["", f"erros ({len(result.erros)}):"]
        for nome, esperado, lido in result.erros[:max_errors]:
            linhas.append(f"  {nome:<28} esperado {esperado:<9} lido {lido}")
        if len(result.erros) > max_errors:
            linhas.append(f"  ... mais {len(result.erros) - max_errors}")

    return "\n".join(linhas)

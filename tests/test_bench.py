from pathlib import Path

import pytest

from lombada.bench import (
    Sample,
    character_error_rate,
    evaluate,
    format_report,
    levenshtein,
    load_samples,
)
from lombada.models import PlateRead

VERDADE = "ABC1D23"


def read(text, confidence=0.9, source="motor"):
    return PlateRead(text=text, confidence=confidence, source=source)


class FakeReader:
    """Devolve leituras canonicas por nome de arquivo."""

    def __init__(self, por_arquivo):
        self.por_arquivo = por_arquivo

    def read_all(self, image):
        return list(self.por_arquivo.get(image, []))


def por_nome(path: Path):
    """`load_image` da bancada: aqui a "imagem" e so o nome do arquivo."""
    return path.name


def amostras(*nomes, truth=VERDADE):
    return [Sample(path=Path(nome), truth=truth) for nome in nomes]


# -- distancia de edicao ---------------------------------------------------


@pytest.mark.parametrize(
    ("a", "b", "esperado"),
    [
        ("ABC1D23", "ABC1D23", 0),
        ("ABC1D23", "ABC1D24", 1),
        ("ABC1D23", "ABC1D2", 1),
        ("", "ABC", 3),
        ("ABC", "", 3),
        ("ABC1D23", "XYZ9K88", 7),
    ],
)
def test_levenshtein(a, b, esperado):
    assert levenshtein(a, b) == esperado


def test_cer_distingue_um_caractere_de_outra_placa():
    """A diferenca que o acerto exato apaga -- e que muda a consequencia."""
    quase = character_error_rate("ABC1D24", VERDADE)
    outra = character_error_rate("XYZ9K88", VERDADE)
    assert quase == pytest.approx(1 / 7)
    assert outra == pytest.approx(1.0)
    assert quase < outra


def test_cer_sem_gabarito():
    assert character_error_rate("", "") == 0.0
    assert character_error_rate("ABC", "") == 1.0


# -- o veredito do ensemble ------------------------------------------------


def test_ganho_quando_o_ensemble_acerta_e_nenhum_motor_sozinho_acerta():
    """A justificativa do ensemble, medida.

    Tres motores erram em posicoes DIFERENTES; nenhum acerta sozinho, e a
    votacao recompoe a placa correta.
    """
    reader = FakeReader(
        {
            "a.jpg": [
                read("XBC1D23", source="m1"),
                read("AXC1D23", source="m2"),
                read("ABX1D23", source="m3"),
            ]
        }
    )
    resultado = evaluate(reader, amostras("a.jpg"), por_nome)

    assert resultado.ensemble_exact == 1
    assert resultado.ensemble_ganhos == 1
    assert resultado.ensemble_regressoes == 0
    assert "se paga" in resultado.veredito()


def test_regressao_quando_dois_motores_errados_afogam_o_que_acertou():
    """O preco do ensemble, medido -- e ele existe de verdade."""
    reader = FakeReader(
        {
            "a.jpg": [
                read("ABC1D23", 0.50, source="certo"),
                read("ABC1D24", 0.90, source="errado1"),
                read("ABC1D24", 0.90, source="errado2"),
            ]
        }
    )
    resultado = evaluate(reader, amostras("a.jpg"), por_nome)

    assert resultado.ensemble_exact == 0
    assert resultado.ensemble_ganhos == 0
    assert resultado.ensemble_regressoes == 1
    assert "PIORA" in resultado.veredito()


def test_veredito_de_empate():
    reader = FakeReader(
        {
            "ganho.jpg": [
                read("XBC1D23", source="m1"),
                read("AXC1D23", source="m2"),
                read("ABX1D23", source="m3"),
            ],
            "perda.jpg": [
                read("ABC1D23", 0.50, source="m1"),
                read("ABC1D24", 0.90, source="m2"),
                read("ABC1D24", 0.90, source="m3"),
            ],
        }
    )
    resultado = evaluate(reader, amostras("ganho.jpg", "perda.jpg"), por_nome)

    assert resultado.ensemble_ganhos == 1
    assert resultado.ensemble_regressoes == 1
    assert "empate" in resultado.veredito()


def test_acerto_de_todos_nao_conta_como_ganho_nem_regressao():
    reader = FakeReader(
        {"a.jpg": [read("ABC1D23", source="m1"), read("ABC1D23", source="m2")]}
    )
    resultado = evaluate(reader, amostras("a.jpg"), por_nome)

    assert resultado.ensemble_exact == 1
    assert resultado.ensemble_ganhos == 0
    assert resultado.ensemble_regressoes == 0


# -- pontuacao por motor ---------------------------------------------------


def test_motor_e_pontuado_uma_vez_por_imagem_mesmo_lendo_varias_vezes():
    """Senao o motor tagarela apareceria melhor so por falar mais."""
    reader = FakeReader(
        {
            "a.jpg": [
                read("ABC1D23", 0.7, source="tagarela"),
                read("ABC1D23", 0.9, source="tagarela"),
                read("ABC1D24", 0.5, source="tagarela"),
                read("ABC1D23", 0.8, source="quieto"),
            ]
        }
    )
    resultado = evaluate(reader, amostras("a.jpg"), por_nome)

    assert resultado.engines["tagarela"].seen == 1
    assert resultado.engines["quieto"].seen == 1
    # A melhor leitura do tagarela (0.9) e a correta.
    assert resultado.engines["tagarela"].exact == 1


def test_taxa_por_motor():
    reader = FakeReader(
        {
            "a.jpg": [read("ABC1D23", source="bom"), read("ABC1D24", source="ruim")],
            "b.jpg": [read("ABC1D23", source="bom"), read("ABC1D25", source="ruim")],
        }
    )
    resultado = evaluate(reader, amostras("a.jpg", "b.jpg"), por_nome)

    assert resultado.engines["bom"].exact_rate == pytest.approx(1.0)
    assert resultado.engines["ruim"].exact_rate == pytest.approx(0.0)
    assert resultado.melhor_motor() == "bom"


# -- cobertura -------------------------------------------------------------


def test_imagem_sem_leitura_conta_como_nao_lida():
    reader = FakeReader({"a.jpg": [read("ABC1D23", source="m")], "b.jpg": []})
    resultado = evaluate(reader, amostras("a.jpg", "b.jpg"), por_nome)

    assert resultado.sem_leitura == 1
    assert resultado.cobertura == pytest.approx(0.5)
    assert resultado.ensemble_cer == pytest.approx(0.5)  # (0.0 + 1.0) / 2


def test_imagem_que_nao_carrega_nao_derruba_a_bancada():
    resultado = evaluate(FakeReader({}), amostras("a.jpg"), lambda _p: None)
    assert resultado.total == 1
    assert resultado.sem_leitura == 1


def test_erros_sao_listados_para_inspecao():
    reader = FakeReader({"a.jpg": [read("XYZ9K88", source="m")]})
    resultado = evaluate(reader, amostras("a.jpg"), por_nome)
    assert resultado.erros == [("a.jpg", "ABC1D23", "XYZ9K88")]


def test_concordancia_media_e_apurada():
    reader = FakeReader(
        {"a.jpg": [read("ABC1D23", source="m1"), read("ABC1D23", source="m2")]}
    )
    assert evaluate(reader, amostras("a.jpg"), por_nome).mean_agreement == 1.0


# -- carga das amostras ----------------------------------------------------


def test_gabarito_vem_do_nome_do_arquivo(tmp_path):
    (tmp_path / "ABC1D23.jpg").write_bytes(b"x")
    (tmp_path / "XYZ9K88_02.png").write_bytes(b"x")
    (tmp_path / "leia-me.txt").write_text("nao e imagem")

    samples = load_samples(tmp_path)

    assert [s.truth for s in samples] == ["ABC1D23", "XYZ9K88"]


def test_gabarito_vem_do_csv(tmp_path):
    labels = tmp_path / "gabarito.csv"
    labels.write_text(
        "arquivo,placa\nfoto1.jpg,abc-1d23\nfoto2.jpg,XYZ9K88\n", encoding="utf-8"
    )
    samples = load_samples(tmp_path, labels)

    assert [(s.path.name, s.truth) for s in samples] == [
        ("foto1.jpg", "ABC1D23"),
        ("foto2.jpg", "XYZ9K88"),
    ]


def test_diretorio_sem_imagem(tmp_path):
    assert load_samples(tmp_path) == []


# -- relatorio -------------------------------------------------------------


def test_relatorio_traz_motores_ensemble_e_veredito():
    reader = FakeReader(
        {"a.jpg": [read("ABC1D23", source="m1"), read("ABC1D24", source="m2")]}
    )
    texto = format_report(evaluate(reader, amostras("a.jpg"), por_nome))

    assert "m1" in texto and "m2" in texto
    assert "ENSEMBLE" in texto
    assert "ganhos do ensemble" in texto


def test_relatorio_sem_amostra():
    from lombada.bench import BenchResult

    assert format_report(BenchResult()) == "nenhuma amostra encontrada"

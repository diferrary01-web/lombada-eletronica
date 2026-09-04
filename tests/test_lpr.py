import pytest

from lombada.config import LprConfig
from lombada.lpr import (
    NullPlateReader,
    build_reader,
    coerce_format,
    is_plausible,
    normalize_plate,
    plate_format,
    vote_plates,
)
from lombada.models import PlateRead


def read(text, confidence=0.9):
    return PlateRead(text=text, confidence=confidence)


# -- votacao posicional ---------------------------------------------------


def test_maioria_corrige_o_caractere_errado_de_uma_leitura():
    """O ganho real do LPR multi-quadro: o erro raramente repete a posicao."""
    voted = vote_plates([read("ABC1D23"), read("ABC1D23"), read("ABCID23")])
    assert voted is not None
    assert voted.text == "ABC1D23"


def test_voto_e_ponderado_pela_confianca():
    """Duas leituras fracas nao derrubam uma leitura forte."""
    voted = vote_plates(
        [read("ABC1D23", 0.95), read("ABC7D23", 0.2), read("ABC8D23", 0.2)]
    )
    assert voted is not None
    assert voted.text == "ABC1D23"


def test_formato_corrige_confusao_que_a_maioria_nao_pega():
    """Todas as leituras erraram igual; o formato ainda salva a posicao."""
    voted = vote_plates([read("ABCID23"), read("ABCID23"), read("ABCID23")])
    assert voted is not None
    assert voted.text == "ABC1D23"


def test_confianca_do_conjunto_e_a_do_caractere_mais_fraco():
    voted = vote_plates([read("ABC1D23"), read("ABC1D24"), read("ABC1D25")])
    assert voted is not None
    # As tres discordam da ultima posicao: essa e a confianca que vale.
    assert voted.confidence == pytest.approx(1 / 3)


def test_leituras_de_tamanho_errado_nao_entram_na_votacao_posicional():
    voted = vote_plates([read("ABC1D23"), read("BC1D23"), read("ABC1D2")])
    assert voted is not None
    assert voted.text == "ABC1D23"


def test_sem_nenhuma_leitura_de_sete_caracteres_devolve_a_mais_confiavel():
    voted = vote_plates([read("ABC1D2", 0.4), read("BC1D23", 0.8)])
    assert voted is not None
    assert voted.text == "BC1D23"


def test_lista_vazia_nao_gera_placa():
    assert vote_plates([]) is None
    assert vote_plates([read("")]) is None


def test_min_votes_exige_repeticao():
    assert vote_plates([read("ABC1D23")], min_votes=2) is None


# -- formato --------------------------------------------------------------


@pytest.mark.parametrize(
    ("entrada", "esperado"),
    [
        ("4BC1D23", "ABC1D23"),  # digito em posicao de letra
        ("ABCO234", "ABC0234"),  # letra em posicao de digito
        ("A8C1D2S", "ABC1D25"),  # os dois tipos de troca na mesma placa
        ("abc-1234", "ABC1234"),  # normalizacao junto
        ("ABC1D23", "ABC1D23"),  # ja correta, nao mexe
    ],
)
def test_coercao_por_posicao(entrada, esperado):
    assert coerce_format(entrada) == esperado


def test_quinta_posicao_nunca_e_coagida():
    """E ela que decide o formato: forcar tipo ali quebraria um dos dois."""
    assert coerce_format("ABC1D23")[4] == "D"  # Mercosul mantem a letra
    assert coerce_format("ABC1234")[4] == "2"  # antiga mantem o digito


def test_texto_de_tamanho_errado_e_so_normalizado():
    assert coerce_format("ab-12") == "AB12"


@pytest.mark.parametrize(
    ("placa", "formato"),
    [
        ("ABC1234", "antiga"),
        ("ABC1D23", "mercosul"),
        ("ABC-1234", "antiga"),
        ("AB12345", None),
        ("ABC12D3", None),
        ("ABC123", None),
        ("", None),
    ],
)
def test_reconhecimento_de_formato(placa, formato):
    assert plate_format(placa) == formato
    assert is_plausible(placa) is (formato is not None)


def test_normalizacao_remove_separadores():
    assert normalize_plate(" abc 1d23 ") == "ABC1D23"


# -- construcao do backend ------------------------------------------------


def test_lpr_desligado_devolve_leitor_nulo():
    reader = build_reader(LprConfig(enabled=False))
    assert isinstance(reader, NullPlateReader)
    assert reader.read(object()) is None


def test_backend_desconhecido_falha_cedo():
    with pytest.raises(ValueError, match="backend de LPR desconhecido"):
        build_reader(LprConfig(backend="inexistente"))

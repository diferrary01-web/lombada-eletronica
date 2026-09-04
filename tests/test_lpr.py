import numpy as np
import pytest

from lombada.config import LprConfig
from lombada.lpr import (
    EnsembleReader,
    NullPlateReader,
    RapidOcrEngine,
    agreement,
    build_engine,
    build_reader,
    coerce_format,
    is_plausible,
    normalize_plate,
    plate_format,
    vote_plates,
)
from lombada.models import PlateRead


def read(text, confidence=0.9, source="motor", weight=1.0, bbox=None):
    return PlateRead(
        text=text, confidence=confidence, source=source, weight=weight, bbox=bbox
    )


def image(height=200, width=400):
    return np.zeros((height, width, 3), dtype=np.uint8)


# -- votacao entre quadros e entre motores --------------------------------


def test_maioria_corrige_o_caractere_errado_de_uma_leitura():
    voted = vote_plates(
        [
            read("ABC1D23", source="a"),
            read("ABC1D23", source="b"),
            read("ABCID23", source="c"),
        ]
    )
    assert voted is not None
    assert voted.text == "ABC1D23"


def test_voto_e_ponderado_pela_confianca():
    voted = vote_plates(
        [
            read("ABC1D23", 0.95, source="a"),
            read("ABC7D23", 0.20, source="b"),
            read("ABC8D23", 0.20, source="c"),
        ]
    )
    assert voted is not None
    assert voted.text == "ABC1D23"


def test_dois_motores_batem_um_motor_que_repetiu_o_mesmo_erro():
    """O ponto do ensemble: nao vence quem fala mais alto.

    Um motor que devolveu leitura em seis quadros nao pode afogar dois motores
    independentes que concordaram entre si -- repetir o mesmo erro seis vezes
    e o modo de falha tipico de um unico modelo, e nao vira evidencia.
    """
    concordes = [read("ABC1D23", 0.60, source="a"), read("ABC1D23", 0.60, source="b")]
    repetido = [read("ABC1D24", 0.50, source="c") for _ in range(6)]

    voted = vote_plates(concordes + repetido)
    assert voted is not None
    assert voted.text == "ABC1D23"


def test_sem_normalizacao_por_motor_a_repeticao_venceria():
    """Guarda o contrapositivo: com um motor so, a maioria bruta decide."""
    voted = vote_plates([read("ABC1D24", 0.50, source="c") for _ in range(6)])
    assert voted is not None
    assert voted.text == "ABC1D24"


def test_peso_do_motor_desempata():
    voted = vote_plates(
        [
            read("ABC1D23", 0.60, source="forte", weight=2.0),
            read("ABC1D24", 0.60, source="fraco", weight=0.5),
        ]
    )
    assert voted is not None
    assert voted.text == "ABC1D23"


def test_formato_corrige_confusao_que_a_maioria_nao_pega():
    voted = vote_plates(
        [read("ABCID23", source=s) for s in ("a", "b", "c")]
    )
    assert voted is not None
    assert voted.text == "ABC1D23"


def test_confianca_do_conjunto_e_a_do_caractere_mais_fraco():
    voted = vote_plates(
        [
            read("ABC1D23", source="a"),
            read("ABC1D24", source="b"),
            read("ABC1D25", source="c"),
        ]
    )
    assert voted is not None
    assert voted.confidence == pytest.approx(1 / 3)


def test_voto_registra_os_motores_que_participaram():
    voted = vote_plates(
        [read("ABC1D23", source="rapidocr"), read("ABC1D23", source="trocr")]
    )
    assert voted is not None
    assert voted.source == "rapidocr+trocr"


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


# -- concordancia entre motores -------------------------------------------


def test_concordancia_total():
    reads = [read("ABC1D23", source="a"), read("ABC1D23", source="b")]
    assert agreement(reads) == pytest.approx(1.0)


def test_concordancia_parcial():
    reads = [
        read("ABC1D23", source="a"),
        read("ABC1D23", source="b"),
        read("XYZ9K88", source="c"),
    ]
    assert agreement(reads) == pytest.approx(2 / 3)


def test_concordancia_ignora_repeticao_do_mesmo_motor():
    """Seis leituras de um motor so continuam sendo um voto."""
    reads = [read("ABC1D23", source="a") for _ in range(6)]
    assert agreement(reads) == pytest.approx(1.0)


def test_concordancia_sem_leitura():
    assert agreement([]) == 0.0


# -- ensemble --------------------------------------------------------------


class FakePrimary:
    name = "primaria"

    def __init__(self, reads):
        self.reads = reads

    def read_all(self, image):  # noqa: ARG002
        return list(self.reads)


class FakeSecondary:
    name = "secundaria"

    def __init__(self, result):
        self.result = result
        self.recebeu = None

    def recognize(self, crop):
        self.recebeu = crop
        return self.result


class MotorQueExplode:
    name = "explode"

    def read_all(self, image):  # noqa: ARG002
        raise RuntimeError("modelo nao carregou")

    def recognize(self, crop):  # noqa: ARG002
        raise RuntimeError("modelo nao carregou")


def test_ensemble_junta_leituras_do_primario_e_dos_secundarios():
    primary = FakePrimary([read("ABC1D23", source="primaria", bbox=(10, 20, 110, 55))])
    secondary = FakeSecondary(read("ABC1D23", source="secundaria"))

    reads = EnsembleReader(primary, [secondary]).read_all(image())

    assert [r.source for r in reads] == ["primaria", "secundaria"]


def test_secundario_recebe_o_recorte_da_placa_nao_o_veiculo_inteiro():
    """E o motivo de existir o primario: TrOCR nao localiza nada sozinho."""
    primary = FakePrimary([read("ABC1D23", source="primaria", bbox=(10, 20, 110, 55))])
    secondary = FakeSecondary(read("ABC1D23", source="secundaria"))

    EnsembleReader(primary, [secondary]).read_all(image(200, 400))

    recorte = secondary.recebeu
    assert recorte is not None
    assert recorte.shape[0] < 200 and recorte.shape[1] < 400


def test_sem_leitura_do_primario_os_secundarios_nao_sao_chamados():
    secondary = FakeSecondary(read("ABC1D23", source="secundaria"))
    reads = EnsembleReader(FakePrimary([]), [secondary]).read_all(image())

    assert reads == []
    assert secondary.recebeu is None


def test_motor_que_explode_nao_derruba_o_ensemble():
    primary = FakePrimary([read("ABC1D23", source="primaria", bbox=(10, 20, 110, 55))])
    reads = EnsembleReader(primary, [MotorQueExplode()]).read_all(image())
    assert [r.source for r in reads] == ["primaria"]


def test_primario_que_explode_devolve_lista_vazia():
    reads = EnsembleReader(MotorQueExplode(), []).read_all(image())
    assert reads == []


def test_nomes_dos_motores():
    ensemble = EnsembleReader(FakePrimary([]), [FakeSecondary(None)])
    assert ensemble.engine_names == ["primaria", "secundaria"]


# -- adaptador do RapidOCR -------------------------------------------------


class FakeRapidResult:
    def __init__(self, txts, scores, boxes):
        self.txts, self.scores, self.boxes = txts, scores, boxes


class FakeRapidEngine:
    def __init__(self, result):
        self.result = result

    def __call__(self, image, **kwargs):  # noqa: ARG002
        return self.result


def caixa(x1, y1, x2, y2):
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


def test_rapidocr_converte_poligono_em_caixa_e_normaliza_texto():
    result = FakeRapidResult(("abc-1d23",), (0.91,), [caixa(10, 20, 110, 55)])
    engine = RapidOcrEngine(LprConfig(), engine=FakeRapidEngine(result))

    (leitura,) = engine.read_all(image())

    assert leitura.text == "ABC1D23"
    assert leitura.bbox == (10.0, 20.0, 110.0, 55.0)
    assert leitura.source == "rapidocr"


def test_rapidocr_descarta_texto_sem_proporcao_de_placa():
    """Adesivo e emblema do modelo tambem sao texto -- e nao sao placa."""
    result = FakeRapidResult(
        ("ABC1D23", "TURBO"),
        (0.95, 0.95),
        [caixa(10, 20, 110, 55), caixa(10, 20, 30, 120)],  # o 2o e alto e estreito
    )
    engine = RapidOcrEngine(LprConfig(), engine=FakeRapidEngine(result))

    textos = [r.text for r in engine.read_all(image())]
    assert textos == ["ABC1D23"]


def test_rapidocr_respeita_a_confianca_minima():
    result = FakeRapidResult(("ABC1D23",), (0.10,), [caixa(10, 20, 110, 55)])
    engine = RapidOcrEngine(
        LprConfig(min_confidence=0.45), engine=FakeRapidEngine(result)
    )
    assert engine.read_all(image()) == []


def test_rapidocr_sem_texto_nenhum():
    engine = RapidOcrEngine(
        LprConfig(), engine=FakeRapidEngine(FakeRapidResult(None, None, None))
    )
    assert engine.read_all(image()) == []


def test_rapidocr_aplica_o_peso_configurado():
    result = FakeRapidResult(("ABC1D23",), (0.91,), [caixa(10, 20, 110, 55)])
    config = LprConfig(engine_weights={"rapidocr": 2.5})
    (leitura,) = RapidOcrEngine(config, engine=FakeRapidEngine(result)).read_all(image())
    assert leitura.weight == pytest.approx(2.5)


# -- construcao ------------------------------------------------------------


def test_lpr_desligado_devolve_leitor_nulo():
    reader = build_reader(LprConfig(enabled=False))
    assert isinstance(reader, NullPlateReader)
    assert reader.read_all(image()) == []


def test_um_motor_so_nao_vira_ensemble():
    reader = build_reader(LprConfig(engines=("stub",)))
    assert isinstance(reader, NullPlateReader)


def test_dois_motores_viram_ensemble():
    reader = build_reader(LprConfig(engines=("rapidocr", "trocr")))
    assert isinstance(reader, EnsembleReader)
    assert reader.engine_names == ["rapidocr", "trocr"]


def test_motor_desconhecido_falha_cedo():
    with pytest.raises(ValueError, match="motor de LPR desconhecido"):
        build_engine("inexistente", LprConfig())


def test_peso_padrao_de_motor_nao_listado():
    assert LprConfig().weight_of("rapidocr") == pytest.approx(1.0)
    assert LprConfig(engine_weights={"trocr": 0.5}).weight_of("trocr") == 0.5


# -- formato ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("entrada", "esperado"),
    [
        ("4BC1D23", "ABC1D23"),
        ("ABCO234", "ABC0234"),
        ("A8C1D2S", "ABC1D25"),
        ("abc-1234", "ABC1234"),
        ("ABC1D23", "ABC1D23"),
    ],
)
def test_coercao_por_posicao(entrada, esperado):
    assert coerce_format(entrada) == esperado


def test_quinta_posicao_nunca_e_coagida():
    assert coerce_format("ABC1D23")[4] == "D"
    assert coerce_format("ABC1234")[4] == "2"


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

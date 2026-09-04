import numpy as np
import pytest

from lombada.probe import (
    CAP_PROP_FOURCC,
    CAP_PROP_FPS,
    mask_url,
    probe_rtsp,
    samples_in_base,
)

URL = "rtsp://admin:segredo@10.0.0.51:554/Streaming/Channels/102"


def frame(width=640, height=360):
    return np.zeros((height, width, 3), dtype=np.uint8)


class FakeCapture:
    def __init__(self, frames, *, opened=True, props=None):
        self.frames = frames
        self._opened = opened
        self.props = props or {}
        self.released = False
        self._index = 0

    def isOpened(self):  # noqa: N802 - assinatura do OpenCV
        return self._opened

    def read(self):
        if self._index >= len(self.frames):
            return False, None
        item = self.frames[self._index]
        self._index += 1
        return True, item

    def get(self, prop):
        return self.props.get(prop, 0.0)

    def release(self):
        self.released = True


def clock_de_passo(step=0.05):
    """Relogio deterministico: cada chamada avanca `step` segundos."""
    estado = {"t": 0.0}

    def clock():
        agora = estado["t"]
        estado["t"] += step
        return agora

    return clock


def sondar(capture, *, frames=5, step=0.05):
    return probe_rtsp(
        URL,
        frames=frames,
        capture_factory=lambda _url: capture,
        clock=clock_de_passo(step),
    )


# -- medicao ---------------------------------------------------------------


def test_mede_o_fps_entregue_em_vez_de_acreditar_no_declarado():
    """A camera declara 25; o relogio diz 20. Vale o que chegou."""
    capture = FakeCapture([frame() for _ in range(5)], props={CAP_PROP_FPS: 25.0})
    result = sondar(capture)

    assert result.ok
    assert result.fps_declarado == pytest.approx(25.0)
    assert result.fps_medido == pytest.approx(20.0)
    assert result.quadros == 5
    assert result.width == 640 and result.height == 360


def test_tempo_ate_o_primeiro_quadro():
    result = sondar(FakeCapture([frame() for _ in range(3)]))
    assert result.primeiro_quadro_s == pytest.approx(0.05)


def test_resolucao_vem_do_quadro_e_nao_da_propriedade():
    """A propriedade mente com frequencia; o quadro decodificado, nao."""
    capture = FakeCapture([frame(2592, 1944) for _ in range(3)])
    result = sondar(capture)
    assert (result.width, result.height) == (2592, 1944)
    assert result.megapixels == pytest.approx(5.04, abs=0.01)


def test_codec_lido_do_fourcc():
    # 'H264' little-endian
    code = ord("H") | (ord("2") << 8) | (ord("6") << 16) | (ord("4") << 24)
    capture = FakeCapture([frame() for _ in range(3)], props={CAP_PROP_FOURCC: code})
    assert sondar(capture).codec == "H264"


# -- avisos ----------------------------------------------------------------


def test_avisa_quando_o_substream_e_na_verdade_o_fluxo_principal():
    """A pegadinha da Dahua: pede-se o secundario e vem 5 MP."""
    result = sondar(FakeCapture([frame(2592, 1944) for _ in range(5)]))
    assert any("MP" in a and "Dahua" in a for a in result.avisos)


def test_nao_avisa_de_resolucao_para_um_substream_de_verdade():
    result = sondar(FakeCapture([frame(640, 360) for _ in range(5)]))
    assert not any("Dahua" in a for a in result.avisos)


def test_avisa_quando_entrega_muito_menos_fps_do_que_declara():
    capture = FakeCapture([frame() for _ in range(5)], props={CAP_PROP_FPS: 25.0})
    result = sondar(capture, step=0.5)  # ~2 fps medidos
    assert any("declara" in a for a in result.avisos)


def test_avisa_que_fps_baixo_nao_mede_velocidade():
    result = sondar(FakeCapture([frame() for _ in range(5)]), step=0.5)
    assert any("pouco para medir velocidade" in a for a in result.avisos)


def test_avisa_demora_ate_o_primeiro_quadro():
    result = sondar(FakeCapture([frame() for _ in range(5)]), step=6.0)
    assert any("primeiro quadro" in a for a in result.avisos)


def test_avisa_fluxo_instavel_com_poucos_quadros():
    result = sondar(FakeCapture([frame(), frame()]), frames=20)
    assert any("instavel" in a for a in result.avisos)


# -- falhas ----------------------------------------------------------------


def test_fluxo_que_nao_abre():
    capture = FakeCapture([], opened=False)
    result = sondar(capture)
    assert not result.ok
    assert "nao foi possivel abrir" in (result.erro or "")
    assert capture.released


def test_fluxo_que_abre_mas_nao_entrega_quadro():
    capture = FakeCapture([])
    result = sondar(capture)
    assert not result.ok
    assert "nao entregou quadro" in (result.erro or "")
    assert capture.released


def test_erro_ao_construir_o_capture_vira_resultado_e_nao_excecao():
    def explode(_url):
        raise RuntimeError("driver ausente")

    result = probe_rtsp(URL, capture_factory=explode)
    assert not result.ok
    assert "driver ausente" in (result.erro or "")


def test_capture_e_liberado_mesmo_no_caminho_feliz():
    capture = FakeCapture([frame() for _ in range(3)])
    sondar(capture)
    assert capture.released


# -- senha -----------------------------------------------------------------


def test_a_senha_nunca_sai_no_resultado():
    result = sondar(FakeCapture([frame() for _ in range(3)]))
    assert "segredo" not in result.url_mascarada
    assert "segredo" not in str(result.to_dict())
    assert result.url_mascarada.startswith("rtsp://admin:***@")


@pytest.mark.parametrize(
    ("entrada", "esperado"),
    [
        ("rtsp://admin:x@1.2.3.4/live", "rtsp://admin:***@1.2.3.4/live"),
        ("rtsp://1.2.3.4/live", "rtsp://1.2.3.4/live"),
        ("sem-esquema", "sem-esquema"),
    ],
)
def test_mascara_de_url(entrada, esperado):
    assert mask_url(entrada) == esperado


# -- a conta que decide se da para medir -----------------------------------


def test_amostras_na_base():
    # 40 km/h = 11,11 m/s; 8 m levam 0,72 s; a 12 fps sao 8,6 amostras.
    assert samples_in_base(12, 8.0, 40) == pytest.approx(8.64, abs=0.01)


def test_amostras_caem_com_a_velocidade():
    devagar = samples_in_base(12, 8.0, 30)
    rapido = samples_in_base(12, 8.0, 90)
    assert devagar > rapido
    assert rapido < 4  # abaixo do minimo: a passagem seria recusada


def test_amostras_com_entrada_degenerada():
    assert samples_in_base(0, 8.0, 40) == 0.0
    assert samples_in_base(12, 8.0, 0) == 0.0

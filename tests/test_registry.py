import pytest
import yaml

from lombada.config import load_config
from lombada.registry import (
    CameraDraft,
    CameraRegistry,
    RegistryError,
    slugify,
)

RTSP = "rtsp://admin:segredo123@10.0.0.51:554/Streaming/Channels/101"
PONTOS = [[320, 900], [1600, 900], [1180, 520], [740, 520]]


@pytest.fixture
def registry(tmp_path):
    return CameraRegistry(path=tmp_path / "config" / "cameras.yaml")


def payload(**kwargs):
    dados = {"rtsp_url": RTSP, "name": "LOMBADA 01", "limit_kmh": 30, "distance_m": 8}
    dados.update(kwargs)
    return dados


# -- cadastro --------------------------------------------------------------


def test_camera_nova_entra_desabilitada_e_sem_calibracao(registry):
    """Sem os 4 pontos nao ha medida -- deixa-la ligada seria mentira."""
    salva = registry.save(CameraDraft.from_payload(payload()))

    assert salva.id == "lombada-01"
    assert not salva.calibrated
    assert salva.to_public()["enabled"] is False


def test_calibrar_liga_a_camera(registry):
    registry.save(CameraDraft.from_payload(payload()))
    salva = registry.calibrate("lombada-01", PONTOS)

    assert salva.calibrated
    assert salva.to_public()["enabled"] is True
    assert salva.image_points == [
        [320.0, 900.0],
        [1600.0, 900.0],
        [1180.0, 520.0],
        [740.0, 520.0],
    ]


def test_id_sai_do_nome_quando_nao_informado(registry):
    salva = registry.save(CameraDraft.from_payload(payload(name="Portaria Fundos 2")))
    assert salva.id == "portaria-fundos-2"


def test_regravar_a_mesma_camera_nao_duplica(registry):
    registry.save(CameraDraft.from_payload(payload()))
    registry.save(CameraDraft.from_payload(payload(name="LOMBADA 01", limit_kmh=40)))

    cameras = registry.cameras()
    assert len(cameras) == 1
    assert cameras[0].limit_kmh == 40


def test_remover(registry):
    registry.save(CameraDraft.from_payload(payload()))
    assert registry.delete("lombada-01") is True
    assert registry.cameras() == []
    assert registry.delete("lombada-01") is False


def test_calibrar_camera_inexistente(registry):
    with pytest.raises(RegistryError, match="desconhecida"):
        registry.calibrate("nao-existe", PONTOS)


# -- validacao -------------------------------------------------------------


def test_url_vazia():
    with pytest.raises(RegistryError, match="informe a URL"):
        CameraDraft.from_payload(payload(rtsp_url=""))


def test_url_com_esquema_errado():
    with pytest.raises(RegistryError, match="rtsp://"):
        CameraDraft.from_payload(payload(rtsp_url="10.0.0.51:554/stream"))


@pytest.mark.parametrize("campo", ["limit_kmh", "distance_m", "capture_fps"])
def test_numero_nao_positivo(campo):
    with pytest.raises(RegistryError, match="maior que zero"):
        CameraDraft.from_payload(payload(**{campo: -1}))


def test_numero_invalido():
    with pytest.raises(RegistryError, match="invalido"):
        CameraDraft.from_payload(payload(limit_kmh="trinta"))


def test_campo_vazio_cai_no_padrao():
    draft = CameraDraft.from_payload(payload(limit_kmh="", capture_fps=""))
    assert draft.limit_kmh == 30.0
    assert draft.capture_fps == 12.0


def test_pontos_em_numero_errado_nao_calibram(registry):
    registry.save(CameraDraft.from_payload(payload()))
    with pytest.raises(RegistryError, match="4 pontos"):
        registry.calibrate("lombada-01", PONTOS[:3])


def test_pontos_malformados(registry):
    registry.save(CameraDraft.from_payload(payload()))
    with pytest.raises(RegistryError, match="malformados"):
        registry.calibrate("lombada-01", ["a", "b", "c", "d"])


# -- senha -----------------------------------------------------------------


def test_a_senha_nunca_volta_para_o_navegador(registry):
    salva = registry.save(CameraDraft.from_payload(payload()))
    publico = salva.to_public()

    assert "segredo123" not in str(publico)
    assert publico["rtsp_url"] == "rtsp://admin:***@10.0.0.51:554/Streaming/Channels/101"


def test_a_senha_e_gravada_no_arquivo_que_o_pipeline_le(registry):
    """O mascaramento e de apresentacao: o pipeline precisa da senha real."""
    registry.save(CameraDraft.from_payload(payload()))
    assert "segredo123" in registry.path.read_text(encoding="utf-8")


def test_o_arquivo_avisa_que_contem_credencial(registry):
    registry.save(CameraDraft.from_payload(payload()))
    assert "credencial" in registry.path.read_text(encoding="utf-8")


# -- integracao com a configuracao real ------------------------------------


def test_o_yaml_gerado_pela_tela_e_carregavel_pela_cli(registry):
    """O ponto do cadastro: a tela edita o arquivo de verdade, nao um espelho."""
    registry.save(CameraDraft.from_payload(payload()))
    registry.calibrate("lombada-01", PONTOS)

    config = load_config(registry.path)
    camera = config.camera("lombada-01")

    assert camera.enabled
    assert camera.limit_kmh == 30.0
    assert camera.base.distance_m == 8.0
    camera.base.homography()  # calibracao utilizavel


def test_camera_sem_calibracao_ainda_gera_yaml_carregavel(registry):
    """Placeholder nos pontos so para o arquivo abrir -- e ela fica desligada."""
    registry.save(CameraDraft.from_payload(payload()))

    config = load_config(registry.path)
    assert config.camera("lombada-01").enabled is False


def test_secoes_alheias_sao_preservadas(registry):
    registry.path.parent.mkdir(parents=True, exist_ok=True)
    registry.path.write_text(
        yaml.safe_dump(
            {
                "site": {"name": "Condominio X"},
                "quality": {"min_r2": 0.95},
                "cameras": [],
            }
        ),
        encoding="utf-8",
    )

    registry.save(CameraDraft.from_payload(payload()))

    document = yaml.safe_load(registry.path.read_text(encoding="utf-8"))
    assert document["site"]["name"] == "Condominio X"
    assert document["quality"]["min_r2"] == 0.95


def test_arquivo_que_nao_e_mapeamento(registry):
    registry.path.parent.mkdir(parents=True, exist_ok=True)
    registry.path.write_text("- uma\n- lista\n", encoding="utf-8")
    with pytest.raises(RegistryError, match="mapeamento"):
        registry.cameras()


def test_registry_sem_arquivo_comeca_vazio(registry):
    assert registry.cameras() == []
    assert registry.get("qualquer") is None


# -- slug ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("entrada", "esperado"),
    [
        ("LOMBADA 01", "lombada-01"),
        ("  Portaria   Fundos ", "portaria-fundos"),
        ("Entrada/Saída #2", "entrada-saida-2"),
        ("!!!", "camera"),
    ],
)
def test_slugify(entrada, esperado):
    assert slugify(entrada) == esperado

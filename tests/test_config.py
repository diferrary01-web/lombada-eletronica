from datetime import time
from textwrap import dedent

import pytest

from lombada.config import ConfigError, Schedule, load_config

YAML = """
site:
  name: Condominio Teste
  timezone: America/Sao_Paulo

detector:
  backend: stub
  confidence: 0.4
  classes: [car, truck]

lpr:
  enabled: false

storage:
  database_path: {db}
  evidence_dir: {ev}
  retention_days: 15

cameras:
  - id: lombada-01
    name: LOMBADA ELETRONICA 01
    rtsp_url: "rtsp://admin:${{CAM_SENHA}}@10.0.0.50:554/Streaming/Channels/101"
    limit_kmh: 30
    capture_fps: 12
    base:
      distance_m: 8.0
      image_points: [[320, 900], [1600, 900], [1180, 520], [740, 520]]
    schedule:
      start: "18:00"
      end: "06:00"
"""


def fill(body, tmp_path):
    """Resolve os marcadores de caminho do YAML de exemplo."""
    return body.format(
        db=(tmp_path / "l.db").as_posix(), ev=(tmp_path / "ev").as_posix()
    )


def write(tmp_path, body=None, raw=None):
    """Grava o YAML: `body` passa pelo `fill`, `raw` vai literal."""
    path = tmp_path / "cameras.yaml"
    text = raw if raw is not None else fill(body or YAML, tmp_path)
    path.write_text(dedent(text), encoding="utf-8")
    return path


def test_carga_completa(tmp_path, monkeypatch):
    monkeypatch.setenv("CAM_SENHA", "s3nh4")
    config = load_config(write(tmp_path))

    assert config.site_name == "Condominio Teste"
    assert config.detector.classes == ("car", "truck")
    assert config.detector.confidence == 0.4
    assert config.lpr.enabled is False
    assert config.storage.retention_days == 15

    camera = config.camera("lombada-01")
    assert camera.limit_kmh == 30
    assert camera.base.distance_m == 8.0
    assert camera.schedule == Schedule(time(18, 0), time(6, 0))


def test_senha_vem_do_ambiente_e_nao_do_arquivo(tmp_path, monkeypatch):
    """O YAML pode ir para o git porque a credencial nao esta nele."""
    monkeypatch.setenv("CAM_SENHA", "s3nh4")
    path = write(tmp_path)
    assert "s3nh4" not in path.read_text(encoding="utf-8")

    camera = load_config(path).camera("lombada-01")
    assert "s3nh4" in camera.rtsp_url


def test_variavel_de_ambiente_faltando_falha_na_carga(tmp_path, monkeypatch):
    monkeypatch.delenv("CAM_SENHA", raising=False)
    with pytest.raises(ConfigError, match="CAM_SENHA"):
        load_config(write(tmp_path))


def test_valor_padrao_na_interpolacao(tmp_path, monkeypatch):
    monkeypatch.delenv("CAM_PORTA", raising=False)
    body = YAML.replace("10.0.0.50:554", "10.0.0.50:${{CAM_PORTA:-554}}")
    monkeypatch.setenv("CAM_SENHA", "x")
    config = load_config(write(tmp_path, body))
    assert ":554/" in config.camera("lombada-01").rtsp_url


def test_calibracao_degenerada_falha_na_carga(tmp_path, monkeypatch):
    monkeypatch.setenv("CAM_SENHA", "x")
    body = YAML.replace(
        "[[320, 900], [1600, 900], [1180, 520], [740, 520]]",
        "[[0, 0], [10, 0], [20, 0], [30, 0]]",
    )
    with pytest.raises(ConfigError):
        load_config(write(tmp_path, body))


def test_ids_repetidos_sao_rejeitados(tmp_path, monkeypatch):
    monkeypatch.setenv("CAM_SENHA", "x")
    formatted = fill(YAML, tmp_path)
    _, camera_block = formatted.split("cameras:\n", 1)
    with pytest.raises(ConfigError, match="repetidos"):
        load_config(write(tmp_path, raw=formatted + camera_block))


def test_chave_desconhecida_e_erro_e_nao_silencio(tmp_path, monkeypatch):
    """Errar o nome de um parametro tem que doer na carga, nao em producao."""
    monkeypatch.setenv("CAM_SENHA", "x")
    body = YAML.replace("confidence: 0.4", "confianca: 0.4")
    with pytest.raises(ConfigError, match="confianca"):
        load_config(write(tmp_path, body))


def test_arquivo_inexistente(tmp_path):
    with pytest.raises(ConfigError, match="nao encontrado"):
        load_config(tmp_path / "nao-existe.yaml")


def test_sem_cameras(tmp_path):
    with pytest.raises(ConfigError, match="nenhuma camera"):
        load_config(write(tmp_path, "site:\n  name: vazio\n"))


# -- janela de operacao ---------------------------------------------------


def test_janela_que_cruza_a_meia_noite():
    noturna = Schedule(time(18, 0), time(6, 0))
    assert noturna.contains(time(23, 30))
    assert noturna.contains(time(3, 0))
    assert noturna.contains(time(18, 0))
    assert not noturna.contains(time(12, 0))
    assert not noturna.contains(time(6, 1))


def test_janela_dentro_do_mesmo_dia():
    diurna = Schedule(time(8, 0), time(18, 0))
    assert diurna.contains(time(12, 0))
    assert not diurna.contains(time(3, 0))


def test_camera_desativada_nunca_esta_ativa(tmp_path, monkeypatch):
    monkeypatch.setenv("CAM_SENHA", "x")
    body = YAML.replace("id: lombada-01", "id: lombada-01\n    enabled: false")
    config = load_config(write(tmp_path, body))
    camera = config.camera("lombada-01")
    assert not camera.active_at(time(23, 0))


def test_camera_desconhecida(tmp_path, monkeypatch):
    monkeypatch.setenv("CAM_SENHA", "x")
    config = load_config(write(tmp_path))
    with pytest.raises(ConfigError, match="camera desconhecida"):
        config.camera("nao-existe")

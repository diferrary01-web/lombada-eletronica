from datetime import datetime, timedelta

import pytest

from lombada.models import Passage, PlateRead
from lombada.storage import PassageStore


def passage(**kwargs):
    params = {
        "camera_id": "lombada-01",
        "track_id": 1,
        "captured_at": datetime(2026, 9, 3, 22, 15, 0),
        "speed_kmh": 48.0,
        "considered_kmh": 41.0,
        "limit_kmh": 30.0,
        "label": "car",
        "plate": PlateRead(text="ABC1D23", confidence=0.87),
        "quality": {"r2": 0.994, "amostras": 9},
    }
    params.update(kwargs)
    return Passage(**params)


@pytest.fixture
def store(tmp_path):
    return PassageStore(tmp_path / "sub" / "lombada.db")


def test_cria_o_banco_e_o_diretorio(tmp_path):
    PassageStore(tmp_path / "nova" / "pasta" / "l.db")
    assert (tmp_path / "nova" / "pasta" / "l.db").exists()


def test_salva_e_recupera(store):
    rowid = store.save(passage())
    assert rowid > 0

    (row,) = store.recent(limit=10)
    assert row["camera_id"] == "lombada-01"
    assert row["speed_kmh"] == pytest.approx(48.0)
    assert row["is_violation"] is True
    assert row["plate"].text == "ABC1D23"
    assert row["quality"]["amostras"] == 9


def test_passagem_dentro_do_limite_nao_e_infracao(store):
    store.save(passage(speed_kmh=34.0, considered_kmh=27.0))
    (row,) = store.recent()
    assert row["is_violation"] is False


def test_filtro_por_infracao(store):
    store.save(passage(track_id=1, speed_kmh=48.0, considered_kmh=41.0))
    store.save(passage(track_id=2, speed_kmh=31.0, considered_kmh=24.0))

    assert len(store.recent()) == 2
    assert len(store.recent(only_violations=True)) == 1


def test_filtro_por_camera(store):
    store.save(passage(camera_id="lombada-01"))
    store.save(passage(camera_id="lombada-02", track_id=2))
    assert len(store.recent(camera_id="lombada-02")) == 1


def test_passagem_sem_placa(store):
    store.save(passage(plate=None))
    (row,) = store.recent()
    assert row["plate"] is None


def test_ordem_e_do_mais_recente_para_o_mais_antigo(store):
    store.save(passage(track_id=1, captured_at=datetime(2026, 9, 3, 20, 0)))
    store.save(passage(track_id=2, captured_at=datetime(2026, 9, 3, 23, 0)))
    assert [r["track_id"] for r in store.recent()] == [2, 1]


def test_resumo_por_camera(store):
    store.save(passage(track_id=1, speed_kmh=40.0, considered_kmh=33.0))
    store.save(passage(track_id=2, speed_kmh=60.0, considered_kmh=53.0))
    store.save(passage(track_id=3, speed_kmh=20.0, considered_kmh=13.0, plate=None))

    (resumo,) = store.stats()
    assert resumo["total"] == 3
    assert resumo["violations"] == 2
    assert resumo["max_kmh"] == pytest.approx(60.0)
    assert resumo["avg_kmh"] == pytest.approx(40.0)
    assert resumo["with_plate"] == 2


def test_retencao_apaga_registro_e_evidencia(tmp_path):
    """Placa e dado pessoal: o prazo tem que apagar o arquivo, nao so a linha."""
    store = PassageStore(tmp_path / "l.db")
    evidence_dir = tmp_path / "evidencias"
    relativo = "lombada-01/2026-07-01/221500_1_overview.jpg"
    arquivo = evidence_dir / relativo
    arquivo.parent.mkdir(parents=True)
    arquivo.write_bytes(b"jpeg")

    antiga = datetime.now() - timedelta(days=40)
    store.save(passage(captured_at=antiga, evidence={"overview": relativo}))
    store.save(passage(track_id=2, captured_at=datetime.now()))

    removidas = store.purge_older_than(30, evidence_dir)

    assert removidas == 1
    assert not arquivo.exists()
    assert [r["track_id"] for r in store.recent()] == [2]


def test_retencao_desligada_nao_apaga_nada(store):
    store.save(passage(captured_at=datetime.now() - timedelta(days=400)))
    assert store.purge_older_than(0) == 0
    assert len(store.recent()) == 1


def test_recorte_temporal(store):
    store.save(passage(track_id=1, captured_at=datetime.now() - timedelta(days=5)))
    store.save(passage(track_id=2, captured_at=datetime.now()))

    recentes = store.recent(since=datetime.now() - timedelta(days=1))
    assert [r["track_id"] for r in recentes] == [2]

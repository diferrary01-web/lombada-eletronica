from lombada.models import Detection
from lombada.track import IouTracker

FPS = 12.0


def det(x, y=400.0, w=100.0, h=80.0, label="car", confidence=0.9):
    return Detection(bbox=(x, y, x + w, y + h), confidence=confidence, label=label)


def test_alvo_rapido_mantem_o_mesmo_id():
    """Regressao do teto de velocidade silencioso.

    120 px por quadro com caixa de 100 px de largura: as caixas cruas de dois
    quadros seguidos NAO se tocam, entao IoU puro daria um ID novo a cada
    quadro e a passagem nunca acumularia amostras suficientes para virar
    medida. O sistema deixaria de ver justamente os carros rapidos.
    """
    tracker = IouTracker()
    ids = []
    for i in range(6):
        tracked = tracker.update([det(100.0 + 120.0 * i)], i / FPS)
        ids.extend(obj.track_id for obj in tracked)

    assert len(ids) == 5  # o primeiro quadro ainda nao atingiu min_hits
    assert len(set(ids)) == 1


def test_dois_veiculos_proximos_nao_trocam_de_id():
    tracker = IouTracker()
    ids_a, ids_b = [], []
    for i in range(6):
        ts = i / FPS
        tracked = tracker.update(
            [det(100.0 + 120.0 * i, 400.0), det(1000.0 + 60.0 * i, 700.0)], ts
        )
        if len(tracked) == 2:
            ids_a.append(tracked[0].track_id)
            ids_b.append(tracked[1].track_id)

    assert len(ids_a) == 5 and len(ids_b) == 5
    assert len(set(ids_a)) == 1
    assert len(set(ids_b)) == 1
    assert set(ids_a) != set(ids_b)


def test_classe_diferente_nao_e_associada_no_portao_de_distancia():
    """O portao por distancia so vale dentro da mesma classe."""
    tracker = IouTracker()
    tracker.update([det(100.0, label="car")], 0.0)
    tracked = tracker.update([det(220.0, label="truck")], 1 / FPS)

    assert tracked == []  # o caminhao e um alvo novo, ainda sem min_hits
    assert tracker.active == 2


def test_alvo_longe_demais_nao_e_capturado_pelo_portao():
    tracker = IouTracker()
    tracker.update([det(100.0)], 0.0)
    tracker.update([det(900.0)], 1 / FPS)  # 800 px, muito alem do portao
    assert tracker.active == 2


def test_alvo_que_some_expira_pelo_tempo():
    tracker = IouTracker(max_age_s=1.0)
    tracker.update([det(100.0)], 0.0)
    tracker.update([det(140.0)], 0.1)
    assert tracker.active == 1

    tracker.update([], 2.0)
    assert tracker.active == 0


def test_forget_descarta_alvo_ja_medido():
    tracker = IouTracker()
    tracker.update([det(100.0)], 0.0)
    tracked = tracker.update([det(140.0)], 0.1)
    assert len(tracked) == 1

    tracker.forget(tracked[0].track_id)
    assert tracker.active == 0


def test_primeiro_quadro_nao_emite_alvo():
    """min_hits evita transformar um falso positivo isolado em passagem."""
    assert IouTracker().update([det(100.0)], 0.0) == []


def test_velocidade_e_estimada_em_pixels_por_segundo():
    tracker = IouTracker()
    tracker.update([det(100.0)], 0.0)
    tracker.update([det(220.0)], 0.5)
    # 120 px em 0,5 s: o proximo quadro e previsto 120 px adiante em 0,5 s.
    tracked = tracker.update([det(340.0)], 1.0)
    assert len(tracked) == 1
    assert tracked[0].hits == 3

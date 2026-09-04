import math

import pytest

from lombada.speed import SpeedError, apply_legal_tolerance, fit_speed

BASE_M = 8.0


def linear_series(v_mps, y0=0.0, n=12, dt=0.05):
    return [(i * dt, y0 + v_mps * i * dt) for i in range(n)]


def test_velocidade_constante_bate_com_a_grandeza_esperada():
    estimate = fit_speed(linear_series(15.0, n=12, dt=0.08), BASE_M)
    assert estimate.speed_kmh == pytest.approx(54.0, rel=1e-6)
    assert estimate.dt == pytest.approx(BASE_M / 15.0, rel=1e-6)
    assert estimate.r2 == pytest.approx(1.0, abs=1e-9)
    assert not estimate.reversed_direction


def test_veiculo_que_desacelera_e_medido_pela_media_na_base():
    """Y(t) = 20t - 3t^2: entra a 20 m/s e freia sobre a lombada.

    A media na base NAO e a velocidade de entrada; e `d / (t_saida - t_entrada)`,
    com os instantes resolvidos na propria parabola.
    """
    samples = [(t / 20, 20 * (t / 20) - 3 * (t / 20) ** 2) for t in range(11)]
    estimate = fit_speed(samples, BASE_M)

    t_exit = (20 - math.sqrt(400 - 4 * 3 * BASE_M)) / (2 * 3)
    esperado = (BASE_M / t_exit) * 3.6

    assert estimate.model == "quadratic"
    assert estimate.speed_kmh == pytest.approx(esperado, rel=1e-6)
    # A media na base fica abaixo da velocidade de entrada (20 m/s = 72 km/h).
    assert estimate.speed_kmh < 72.0


def test_quadratico_descreve_melhor_quem_freia_do_que_a_reta():
    """Justificativa do ajuste quadratico, medida no mesmo dado sintetico."""
    import numpy as np

    samples = [(t / 20, 20 * (t / 20) - 3 * (t / 20) ** 2) for t in range(11)]
    t = np.array([s[0] for s in samples])
    y = np.array([s[1] for s in samples])

    def r2(degree):
        pred = np.polyval(np.polyfit(t, y, degree), t)
        return 1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)

    assert r2(2) > r2(1)
    assert fit_speed(samples, BASE_M).model == "quadratic"


def test_velocidade_constante_nao_sofre_cancelacao_catastrofica():
    """Regressao: com `a` ~ 1e-14 a Bhaskara direta erra por ordens de grandeza.

    O ajuste de grau 2 sobre um movimento uniforme produz curvatura numerica
    residual. Se as raizes forem resolvidas ingenuamente, o cruzamento sai
    absurdo e a velocidade vai junto -- sem nenhuma excecao para avisar.
    """
    for v_mps in (5.0, 10.0, 15.0, 22.0, 30.0):
        # 20 amostras cobrindo 12 m, para que a base de 8 m caiba na janela
        # observada em qualquer das velocidades.
        samples = linear_series(v_mps, n=20, dt=12.0 / (v_mps * 19))
        estimate = fit_speed(samples, BASE_M)
        assert estimate.speed_kmh == pytest.approx(v_mps * 3.6, rel=1e-6)


def test_sentido_invertido_mede_a_mesma_velocidade():
    samples = [(i * 0.05, 9.0 - 15.0 * i * 0.05) for i in range(15)]
    estimate = fit_speed(samples, BASE_M)
    assert estimate.speed_kmh == pytest.approx(54.0, rel=1e-6)
    assert estimate.reversed_direction
    assert estimate.t_entry < estimate.t_exit


def test_cruzamentos_saem_do_ajuste_nao_do_quadro_mais_proximo():
    """Com amostragem grossa a resposta ainda e exata, porque a raiz e resolvida."""
    grosso = fit_speed(linear_series(15.0, n=6, dt=0.2), BASE_M)
    fino = fit_speed(linear_series(15.0, n=40, dt=0.03), BASE_M)
    assert grosso.speed_kmh == pytest.approx(fino.speed_kmh, rel=1e-6)


def test_amostras_de_menos_sao_recusadas():
    with pytest.raises(SpeedError, match="insuficientes"):
        fit_speed(linear_series(15.0, n=3, dt=0.1), BASE_M, min_samples=4)


def test_veiculo_parado_nao_gera_medida():
    parado = [(i * 0.1, 4.0) for i in range(10)]
    with pytest.raises(SpeedError, match="deslocamento"):
        fit_speed(parado, BASE_M)


def test_faixa_observada_curta_demais_e_recusada():
    """So viu de Y=0 a Y=3 numa base de 8 m: o resto seria extrapolacao."""
    curto = [(i * 0.05, 0.0 + 6.0 * i * 0.05) for i in range(11)]
    with pytest.raises(SpeedError, match="extrapolados"):
        fit_speed(curto, BASE_M, max_extrapolation_m=2.0)


def test_serie_erratica_e_recusada_pelo_r2():
    ruido = [(0.0, 0.0), (0.1, 6.0), (0.2, 1.0), (0.3, 7.5), (0.4, 2.0), (0.5, 9.0)]
    with pytest.raises(SpeedError, match="ajuste ruim"):
        fit_speed(ruido, BASE_M, min_r2=0.9)


def test_velocidade_absurda_e_recusada():
    with pytest.raises(SpeedError, match="fora de faixa"):
        fit_speed(linear_series(120.0, n=12, dt=0.005), BASE_M, max_speed_kmh=250.0)


def test_timestamps_repetidos_nao_quebram_o_ajuste():
    samples = linear_series(15.0, n=12, dt=0.08)
    duplicado = samples + [samples[3], samples[7]]
    estimate = fit_speed(duplicado, BASE_M)
    assert estimate.n_samples == len(samples)
    assert estimate.speed_kmh == pytest.approx(54.0, rel=1e-6)


def test_amostras_fora_de_ordem_sao_ordenadas():
    samples = linear_series(15.0, n=12, dt=0.08)
    embaralhado = samples[6:] + samples[:6]
    assert fit_speed(embaralhado, BASE_M).speed_kmh == pytest.approx(54.0, rel=1e-6)


@pytest.mark.parametrize(
    ("medido", "limite", "esperado"),
    [
        (40.0, 30.0, 33.0),
        (36.0, 30.0, 29.0),  # cai para dentro do limite depois da tolerancia
        (5.0, 30.0, 0.0),
        (130.0, 120.0, 124.0),  # acima de 100 km/h a tolerancia vira 5%
    ],
)
def test_tolerancia_legal(medido, limite, esperado):
    assert apply_legal_tolerance(medido, limite) == pytest.approx(esperado)


def test_distancia_invalida():
    with pytest.raises(SpeedError, match="distance_m"):
        fit_speed(linear_series(15.0), 0.0)

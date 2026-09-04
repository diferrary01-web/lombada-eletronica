import math

import pytest

from lombada.geometry import (
    BaseGeometry,
    CalibrationError,
    contact_point,
    image_to_world,
)

# Trapezio tipico de camera olhando a via em perspectiva: a linha de saida
# (mais longe) aparece mais estreita e mais alta na imagem.
IMAGE_POINTS = ((320.0, 900.0), (1600.0, 900.0), (1180.0, 520.0), (740.0, 520.0))


def base(**kwargs):
    params = {"image_points": IMAGE_POINTS, "distance_m": 8.0}
    params.update(kwargs)
    return BaseGeometry(**params)


def test_cantos_voltam_exatamente_nos_pontos_do_mundo():
    geometry = base()
    homography = geometry.homography()
    for image_point, world_point in zip(
        geometry.image_points, geometry.world_points, strict=True
    ):
        got = image_to_world(homography, image_point)
        assert got == pytest.approx(world_point, abs=1e-9)


def test_largura_suposta_da_via_nao_altera_a_coordenada_longitudinal():
    """A propriedade que dispensa medir a largura da via em campo.

    Dobrar a largura suposta escala o alvo apenas em X, entao a homografia
    resultante e `diag(k,1,1) @ H`: X dobra, Y fica identico. Como so Y entra
    no calculo de velocidade, errar a largura nao contamina a medida.
    """
    narrow = base(lane_width_m=3.5).homography()
    wide = base(lane_width_m=7.0).homography()

    for pixel in ((640.0, 800.0), (1000.0, 700.0), (900.0, 560.0)):
        x_narrow, y_narrow = image_to_world(narrow, pixel)
        x_wide, y_wide = image_to_world(wide, pixel)
        assert y_wide == pytest.approx(y_narrow, rel=1e-9, abs=1e-9)
        assert x_wide == pytest.approx(2.0 * x_narrow, rel=1e-9, abs=1e-9)


def test_ponto_medio_da_base_cai_perto_do_meio_em_metros():
    homography = base().homography()
    entrada = image_to_world(homography, (960.0, 900.0))
    saida = image_to_world(homography, (960.0, 520.0))
    assert entrada[1] == pytest.approx(0.0, abs=1e-9)
    assert saida[1] == pytest.approx(8.0, abs=1e-9)
    # Em perspectiva o meio da IMAGEM nao e o meio da via: fica mais longe.
    meio = image_to_world(homography, (960.0, 710.0))
    assert 0.0 < meio[1] < 8.0


def test_ponto_de_contato_e_o_centro_da_aresta_inferior():
    assert contact_point((100.0, 200.0, 300.0, 400.0)) == (200.0, 400.0)


def test_distancia_nao_positiva_e_rejeitada():
    with pytest.raises(CalibrationError):
        base(distance_m=0.0)


def test_pontos_colineares_sao_rejeitados():
    colineares = ((0.0, 0.0), (10.0, 0.0), (20.0, 0.0), (30.0, 0.0))
    with pytest.raises(CalibrationError):
        BaseGeometry(image_points=colineares, distance_m=8.0).homography()


def test_numero_de_pontos_diferente_de_quatro_e_rejeitado():
    tres_pontos = IMAGE_POINTS[:3]
    with pytest.raises(CalibrationError):
        BaseGeometry(image_points=tres_pontos, distance_m=8.0)  # type: ignore[arg-type]


def test_ponto_na_linha_de_fuga_nao_projeta():
    """Na linha do horizonte o denominador zera; melhor erro que numero falso."""
    import numpy as np

    degenerate = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
    with pytest.raises(CalibrationError):
        image_to_world(degenerate, (960.0, 700.0))


def test_homografia_e_finita_e_normalizada():
    homography = base().homography()
    assert homography.shape == (3, 3)
    assert homography[2, 2] == pytest.approx(1.0)
    assert all(math.isfinite(v) for v in homography.ravel())

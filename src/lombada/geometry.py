"""Homografia imagem->mundo para a base de medicao da lombada.

A calibracao usada aqui e a mesma que um radar de faixa dupla exige em campo:
quatro pontos na imagem marcando o quadrilatero da base, e a distancia real
entre as duas linhas transversais. Nao e preciso medir a largura da via.

Justificativa (importante, e contra-intuitivo): se a largura suposta for
`k` vezes a real, o alvo do mundo fica escalado apenas em X, entao a
homografia obtida e `H' = diag(k, 1, 1) @ H`. A coordenada LONGITUDINAL Y --
a unica que entra no calculo de velocidade -- sai identica. Errar a largura
distorce X, nunca Y.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

Point = tuple[float, float]


class CalibrationError(ValueError):
    """A calibracao fornecida nao gera uma homografia utilizavel."""


@dataclass(frozen=True)
class BaseGeometry:
    """Base de medicao: duas linhas transversais separadas por `distance_m`.

    `image_points` sao os 4 cantos NESTA ordem, olhando a imagem:
        0 = linha de entrada, lado esquerdo
        1 = linha de entrada, lado direito
        2 = linha de saida,   lado direito
        3 = linha de saida,   lado esquerdo

    O sentido de trafego vai de Y=0 (entrada) para Y=distance_m (saida).
    """

    image_points: tuple[Point, Point, Point, Point]
    distance_m: float
    lane_width_m: float = 3.5

    def __post_init__(self) -> None:
        if len(self.image_points) != 4:
            raise CalibrationError("sao necessarios exatamente 4 pontos de imagem")
        if self.distance_m <= 0:
            raise CalibrationError("distance_m deve ser positiva")
        if self.lane_width_m <= 0:
            raise CalibrationError("lane_width_m deve ser positiva")

    @property
    def world_points(self) -> tuple[Point, Point, Point, Point]:
        half = self.lane_width_m / 2.0
        d = self.distance_m
        return ((-half, 0.0), (half, 0.0), (half, d), (-half, d))

    def homography(self) -> np.ndarray:
        return compute_homography(self.image_points, self.world_points)


def compute_homography(
    src: tuple[Point, ...], dst: tuple[Point, ...]
) -> np.ndarray:
    """DLT de 4 pontos. Devolve H 3x3 com H[2,2] == 1, tal que dst ~ H @ src."""
    if len(src) != 4 or len(dst) != 4:
        raise CalibrationError("compute_homography espera 4 pares de pontos")

    a = np.zeros((8, 8), dtype=np.float64)
    b = np.zeros(8, dtype=np.float64)
    for i, ((x, y), (u, v)) in enumerate(zip(src, dst)):
        a[2 * i] = (x, y, 1, 0, 0, 0, -u * x, -u * y)
        b[2 * i] = u
        a[2 * i + 1] = (0, 0, 0, x, y, 1, -v * x, -v * y)
        b[2 * i + 1] = v

    try:
        h = np.linalg.solve(a, b)
    except np.linalg.LinAlgError as exc:  # pontos colineares/degenerados
        raise CalibrationError(f"pontos de calibracao degenerados: {exc}") from exc

    matrix = np.append(h, 1.0).reshape(3, 3)
    if not np.isfinite(matrix).all():
        raise CalibrationError("homografia nao finita")
    return matrix


def image_to_world(h: np.ndarray, point: Point) -> Point:
    """Projeta um ponto da imagem no plano da via. Devolve (X, Y) em metros."""
    x, y = point
    vec = h @ np.array([x, y, 1.0], dtype=np.float64)
    w = vec[2]
    if abs(w) < 1e-12:
        raise CalibrationError("ponto projetado no infinito (linha do horizonte)")
    return float(vec[0] / w), float(vec[1] / w)


def contact_point(bbox: tuple[float, float, float, float]) -> Point:
    """Ponto do veiculo que toca o solo: centro da aresta inferior da bbox.

    Usar o centro da bbox introduz erro sistematico -- a altura do veiculo
    projeta o centroide para frente ou para tras conforme a perspectiva.
    """
    x1, _y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, y2)

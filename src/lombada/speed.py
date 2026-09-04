"""Estimativa de velocidade na base de medicao.

O metodo e o de "tempo entre duas linhas": mede-se o instante em que o ponto
de contato do veiculo cruza Y=0 e Y=distance_m, e a velocidade e a media na
base. E a mesma grandeza que a norma metrologica define para radar de laco
duplo -- media no trecho, nao instantanea.

Duas escolhas que importam:

1. **Ajuste quadratico, nao linear.** Sobre uma lombada o veiculo desacelera
   de verdade; forcar reta faz o residuo comer a medida. Em dados reais de
   campo o ajuste linear ficou em R^2 ~0.86 enquanto o quadratico passou de
   0.99, e a dispersao da velocidade caiu de +-13.7% para +-5.2%.
2. **Cruzamentos resolvidos no ajuste, nao no quadro mais proximo.** Amostrar
   a 12 fps a 50 km/h da 1.2 m entre quadros; arredondar para o quadro mais
   proximo joga fora ate 4% da medida. Resolver a raiz recupera isso.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

MPS_TO_KMH = 3.6


@dataclass(frozen=True)
class SpeedEstimate:
    """Resultado de uma passagem medida."""

    speed_kmh: float
    t_entry: float
    t_exit: float
    r2: float
    model: str
    n_samples: int
    residual_std_m: float
    extrapolated_m: float
    reversed_direction: bool

    @property
    def dt(self) -> float:
        return abs(self.t_exit - self.t_entry)


class SpeedError(ValueError):
    """A serie de posicoes nao permite uma medida confiavel."""


def fit_speed(
    samples: Sequence[tuple[float, float]],
    distance_m: float,
    *,
    min_samples: int = 4,
    min_r2: float = 0.90,
    max_extrapolation_m: float = 2.0,
    max_speed_kmh: float = 250.0,
) -> SpeedEstimate:
    """Ajusta Y(t) e devolve a velocidade media na base.

    `samples` sao pares (timestamp_s, Y_metros) do ponto de contato ja
    projetado pelo `geometry.image_to_world`. Levanta `SpeedError` quando a
    passagem nao atende aos criterios de qualidade -- e melhor nao medir do
    que medir errado.
    """
    if distance_m <= 0:
        raise SpeedError("distance_m deve ser positiva")

    pts = _clean(samples)
    if len(pts) < min_samples:
        raise SpeedError(f"amostras insuficientes: {len(pts)} < {min_samples}")

    t = np.array([p[0] for p in pts], dtype=np.float64)
    y = np.array([p[1] for p in pts], dtype=np.float64)

    t0 = float(t[0])
    t_rel = t - t0

    span = float(y[-1] - y[0])
    if abs(span) < 1e-6:
        raise SpeedError("veiculo sem deslocamento longitudinal observavel")
    direction = math.copysign(1.0, span)

    degree = 2 if len(pts) >= 4 else 1
    coeffs, r2, residual_std = _polyfit(t_rel, y, degree)
    model = "quadratic" if degree == 2 else "linear"

    # Quadratico com curvatura absurda costuma ser ruido; cai para reta.
    if degree == 2 and r2 < min_r2:
        lin_coeffs, lin_r2, lin_std = _polyfit(t_rel, y, 1)
        if lin_r2 > r2:
            coeffs, r2, residual_std, model = lin_coeffs, lin_r2, lin_std, "linear"

    if r2 < min_r2:
        raise SpeedError(f"ajuste ruim: R2={r2:.3f} < {min_r2}")

    t_lo, t_hi = float(t_rel[0]), float(t_rel[-1])
    t_at_0 = _solve_crossing(coeffs, 0.0, t_lo, t_hi, direction)
    t_at_d = _solve_crossing(coeffs, distance_m, t_lo, t_hi, direction)
    if t_at_0 is None or t_at_d is None:
        raise SpeedError("nao foi possivel resolver o cruzamento das linhas")

    dt = abs(t_at_d - t_at_0)
    if dt < 1e-6:
        raise SpeedError("intervalo entre linhas degenerado")

    y_min, y_max = float(y.min()), float(y.max())
    extrapolated = max(0.0, y_min - 0.0) + max(0.0, distance_m - y_max)
    if extrapolated > max_extrapolation_m:
        raise SpeedError(
            f"faixa observada cobre pouco da base: {extrapolated:.2f} m extrapolados"
        )

    speed_kmh = (distance_m / dt) * MPS_TO_KMH
    if not math.isfinite(speed_kmh) or speed_kmh <= 0 or speed_kmh > max_speed_kmh:
        raise SpeedError(f"velocidade fora de faixa plausivel: {speed_kmh:.1f} km/h")

    t_entry, t_exit = (t_at_0, t_at_d) if direction > 0 else (t_at_d, t_at_0)
    return SpeedEstimate(
        speed_kmh=speed_kmh,
        t_entry=t0 + t_entry,
        t_exit=t0 + t_exit,
        r2=r2,
        model=model,
        n_samples=len(pts),
        residual_std_m=residual_std,
        extrapolated_m=extrapolated,
        reversed_direction=direction < 0,
    )


def apply_legal_tolerance(
    measured_kmh: float, limit_kmh: float, *, tolerance_kmh: float = 7.0
) -> float:
    """Velocidade considerada apos a tolerancia do instrumento.

    A pratica brasileira desconta 7 km/h para limites ate 100 km/h e 5% acima
    disso. Numa lombada (limite tipico 30 km/h) vale sempre a faixa fixa.

    ATENCAO: isto reproduz a aritmetica da tolerancia; NAO torna a medida
    valida para autuacao. Ver a secao "Limites legais" do README.
    """
    tolerance = tolerance_kmh if limit_kmh <= 100 else limit_kmh * 0.05
    return max(0.0, measured_kmh - tolerance)


def _clean(samples: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    """Ordena por tempo e remove timestamps repetidos (mantem o primeiro)."""
    ordered = sorted(samples, key=lambda p: p[0])
    out: list[tuple[float, float]] = []
    for ts, y in ordered:
        if not (math.isfinite(ts) and math.isfinite(y)):
            continue
        if out and abs(ts - out[-1][0]) < 1e-9:
            continue
        out.append((float(ts), float(y)))
    return out


def _polyfit(
    t: np.ndarray, y: np.ndarray, degree: int
) -> tuple[np.ndarray, float, float]:
    coeffs = np.polyfit(t, y, degree)
    pred = np.polyval(coeffs, t)
    residuals = y - pred
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 if ss_tot < 1e-12 else 1.0 - ss_res / ss_tot
    dof = max(1, len(y) - (degree + 1))
    residual_std = math.sqrt(ss_res / dof)
    return coeffs, r2, residual_std


def _solve_crossing(
    coeffs: np.ndarray,
    y_target: float,
    t_lo: float,
    t_hi: float,
    direction: float,
) -> float | None:
    """Instante em que a curva ajustada cruza `y_target`.

    Havendo duas raizes (a parabola sobe e desce), fica a que tem a derivada
    no mesmo sentido do trajeto observado -- a outra e o ramo espelhado, que
    nunca aconteceu.

    A formula usada e a estavel (`q = -(b + sinal(b)*sqrt(D))/2`, raizes `q/a`
    e `offset/q`), nao a de Bhaskara direta. Isso importa muito aqui: veiculo
    em velocidade constante da um ajuste com `a` na ordem de 1e-14, e entao
    `-b + sqrt(D)` e a subtracao de dois numeros quase iguais -- a cancelacao
    catastrofica come todos os digitos significativos e a velocidade sai errada
    por ordens de grandeza, sem levantar erro nenhum.
    """
    if len(coeffs) == 2:
        b, c = float(coeffs[0]), float(coeffs[1])
        if abs(b) < 1e-12:
            return None
        return (y_target - c) / b

    a, b, c = (float(v) for v in coeffs)
    offset = c - y_target
    window = max(abs(t_hi - t_lo), 1e-6)

    # Curvatura desprezivel dentro da janela observada: tratar como reta.
    if abs(a) * window <= 1e-9 * abs(b):
        if abs(b) < 1e-12:
            return None
        return -offset / b

    disc = b * b - 4.0 * a * offset
    if disc < 0:
        return None
    sq = math.sqrt(disc)
    q = -0.5 * (b + math.copysign(sq, b if b != 0.0 else 1.0))

    roots = [q / a]
    if abs(q) > 1e-300:
        roots.append(offset / q)

    forward = [r for r in roots if math.copysign(1.0, 2.0 * a * r + b) == direction]
    candidates = forward or roots
    return min(candidates, key=lambda r: _distance_to_window(r, t_lo, t_hi))


def _distance_to_window(value: float, lo: float, hi: float) -> float:
    if value < lo:
        return lo - value
    if value > hi:
        return value - hi
    return 0.0

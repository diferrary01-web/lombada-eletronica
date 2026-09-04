"""Guardas de ambiente e de declaracao de dependencia.

Estes testes nao exercitam logica do sistema: eles impedem a volta de um
acidente concreto. Em 2026-09-04 o OpenCV 5.0 entrou neste projeto apesar de
o extra `video` ja pinar `<5`, porque `ultralytics` e `rapidocr` dependem da
distribuicao **`opencv-python`** enquanto o pin estava em
**`opencv-python-headless`** -- dois nomes distintos, que instalam o mesmo
modulo `cv2` e que o pip trata como pacotes sem relacao. O mesmo caminho ja
tinha quebrado a decodificacao RTSP de um parque de cameras meses antes.

O teste de pyproject roda em qualquer lugar, inclusive na CI sem OpenCV. O de
runtime so roda onde o `cv2` existe.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"

# Distribuicoes que instalam o modulo `cv2`. Pinar uma nao restringe a outra.
OPENCV_DISTRIBUTIONS = (
    "opencv-python",
    "opencv-python-headless",
    "opencv-contrib-python",
    "opencv-contrib-python-headless",
)


def requisitos() -> dict[str, list[str]]:
    data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    project = data["project"]
    grupos = {"__main__": project.get("dependencies", [])}
    grupos.update(project.get("optional-dependencies", {}))
    return grupos


def normalizar(nome: str) -> str:
    return re.sub(r"[-_.]+", "-", nome).lower()


def nome_da_dependencia(requisito: str) -> str:
    return normalizar(re.split(r"[<>=!~\[;\s]", requisito.strip(), maxsplit=1)[0])


def test_todo_extra_que_puxa_opencv_poe_teto_de_versao():
    """A regressao concreta: pin no nome errado nao segura nada."""
    faltando = []
    for grupo, requisitos_do_grupo in requisitos().items():
        for requisito in requisitos_do_grupo:
            if nome_da_dependencia(requisito) not in OPENCV_DISTRIBUTIONS:
                continue
            if "<5" not in requisito.replace(" ", ""):
                faltando.append(f"{grupo}: {requisito}")

    assert not faltando, (
        "requisito de OpenCV sem teto `<5` -- o OpenCV 5.0 quebra a "
        f"decodificacao RTSP: {faltando}"
    )


@pytest.mark.parametrize("extra", ["detect", "lpr"])
def test_extras_que_arrastam_opencv_pinam_a_variante_nao_headless(extra):
    """`ultralytics` e `rapidocr` dependem de `opencv-python`, nao da headless.

    Sem repetir o pin nessa variante, o resolvedor fica livre para trazer a
    5.x mesmo com o extra `video` pinado.
    """
    nomes = {nome_da_dependencia(r) for r in requisitos()[extra]}
    assert "opencv-python" in nomes


def test_opencv_instalado_nao_e_a_serie_5():
    """Guarda de runtime: no ambiente onde ha `cv2`, ele precisa ser 4.x."""
    cv2 = pytest.importorskip("cv2", reason="OpenCV nao instalado neste ambiente")
    major = int(cv2.__version__.split(".")[0])
    assert major < 5, (
        f"OpenCV {cv2.__version__} instalado. A serie 5.x quebra a leitura "
        "RTSP usada pela captura; fixe `<5` nas DUAS distribuicoes "
        "(`opencv-python` e `opencv-python-headless`)."
    )

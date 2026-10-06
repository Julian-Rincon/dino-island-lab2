"""Secciones 6, 7 y 9 del PDF: contenido del repositorio y ausencia de información sensible."""
import re
import subprocess

import pytest

from conftest import RAIZ

ARCHIVOS = subprocess.run(["git", "ls-files"], cwd=RAIZ, capture_output=True, text=True).stdout.split()


@pytest.mark.parametrize("ruta", [
    "1_generador_nombres/entrenar.py",                 # generador de nombres
    "2_ollama_sagemaker/on-start.sh",                  # configuración de Ollama
    "2_ollama_sagemaker/identidad_ollama.ipynb",
    "3_imagen/colab_imagen.ipynb",                     # código de generación de imágenes
    "3_imagen/servidor_imagen.py",
    "4_app/lambda/lambda_function.py",                 # archivos de la aplicación
    "4_app/lambda/index.html",
    "4_app/deploy.sh",
    "README.md",
    "ENTREGA.md",
])
def test_entregable_en_repo(ruta):
    assert ruta in ARCHIVOS


@pytest.mark.parametrize("evidencia", [
    "1_generador_nombres/resultados/resumen.json",      # arquitectura y configuración
    "1_generador_nombres/resultados/curvas_perdida.png",  # curvas
    "1_generador_nombres/resultados/muestreo.md",       # configuraciones de muestreo
])
def test_evidencias_experimentales(evidencia):
    assert evidencia in ARCHIVOS


def test_readme_documenta_componentes():
    r = (RAIZ / "README.md").read_text()
    for s in ("Qué recibe y devuelve cada componente", "deploy.sh", "on-start.sh", "Diez nombres", "Sitio"):
        assert s in r


PATRONES = [r"AKIA[0-9A-Z]{16}", r"ASIA[0-9A-Z]{16}", r"aws_secret_access_key\s*=", r"aws_session_token\s*=",
            r"-----BEGIN [A-Z ]*PRIVATE KEY-----", r"\bolm_[A-Za-z0-9]{20,}", r"\bhf_[A-Za-z0-9]{30,}",
            r"\bghp_[A-Za-z0-9]{30,}", r"\b2[a-zA-Z0-9]{26}_[a-zA-Z0-9]{20,}\b"]


def test_sin_credenciales_ni_tokens():
    hallazgos = []
    for f in ARCHIVOS:
        p = RAIZ / f
        if p.suffix in (".png", ".pt", ".npz") or f.startswith("pruebas/capturas"):
            continue
        texto = p.read_text(errors="ignore")
        hallazgos += [(f, pat) for pat in PATRONES if re.search(pat, texto)]
    assert not hallazgos, hallazgos


def test_sin_archivos_sensibles():
    assert not [f for f in ARCHIVOS if re.search(r"(\.pem|\.env|credentials|\.venv)", f)]

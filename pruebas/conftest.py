"""Configuración común de las pruebas de entrega."""
import os
import re
import unicodedata
from pathlib import Path

import pandas as pd
import pytest
import requests

RAIZ = Path(__file__).resolve().parent.parent
SITIO = os.environ.get("SITIO", "https://zk6dst5eom6dhbjfcivxejcpn40iakbw.lambda-url.us-east-1.on.aws").rstrip("/")
DINO_INICIAL = "limedsaurus-10553b"
CLAVES_IDENTIDAD = ["significado_nombre", "periodo", "tamano", "apariencia", "habitat", "alimentacion",
                    "comportamiento", "rasgo_distintivo", "descripcion", "prompt_imagen"]


def normalizar(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z]", "", s.lower())


@pytest.fixture(scope="session")
def dataset():
    """Nombres reales del dinos.csv del laboratorio, normalizados de forma independiente."""
    df = pd.read_csv(RAIZ / "1_generador_nombres" / "data" / "dinos.csv", header=None)
    return {normalizar(n) for n in df[0]}


@pytest.fixture(scope="session")
def api():
    s = requests.Session()

    def llamar(metodo, ruta, json=None, timeout=300):
        return s.request(metodo, SITIO + ruta, json=json, timeout=timeout)
    return llamar

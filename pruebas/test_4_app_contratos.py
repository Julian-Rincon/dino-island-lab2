"""Sección 5 y 9 del PDF: integración y chat, verificando el contrato de cada ruta del README."""
import io
import time

import pytest
from PIL import Image

from conftest import CLAVES_IDENTIDAD, DINO_INICIAL, normalizar


# --- Rutas y contratos -----------------------------------------------------
def test_sitio_accesible(api):
    r = api("GET", "/")
    assert r.status_code == 200 and "text/html" in r.headers["content-type"]
    assert "Nuevo dinosaurio" in r.text


def test_contrato_info(api):
    j = api("GET", "/api/info").json()
    assert {"arquitectura", "parametros", "val_loss", "dataset", "muestreo"} <= j["modelo_nombres"].keys()
    assert j["modelo_ollama"] == "gemma4:e2b"


def test_contrato_nombre_sin_reentrenar(api, dataset):
    t0 = time.time()
    nombres = [api("POST", "/api/nombre", {}).json() for _ in range(5)]
    # 5 nombres en pocos segundos: solo muestreo del modelo ya entrenado
    assert time.time() - t0 < 30
    for j in nombres:
        assert set(j) == {"nombre", "modelo"}
        assert normalizar(j["nombre"]) not in dataset
    assert len({j["nombre"] for j in nombres}) == 5


def test_contrato_dino_actual(api):
    d = api("GET", "/api/dino/actual").json()
    assert d["id"] == DINO_INICIAL
    assert set(CLAVES_IDENTIDAD) <= d["identidad"].keys()
    assert d["imagen"] == f"/img/{DINO_INICIAL}.png"
    assert {"modelo_nombres", "modelo_ollama", "modelo_imagen"} <= d.keys()


def test_contrato_imagen_png(api):
    r = api("GET", f"/img/{DINO_INICIAL}.png")
    assert r.status_code == 200 and r.headers["content-type"] == "image/png"
    assert Image.open(io.BytesIO(r.content)).size == (512, 512)


def test_imagen_se_genera_una_vez_y_se_reutiliza(api):
    t0 = time.time()
    j = api("POST", "/api/imagen", {"id": DINO_INICIAL}).json()
    assert j == {"id": DINO_INICIAL, "imagen_url": f"/img/{DINO_INICIAL}.png", "reutilizada": True}
    assert time.time() - t0 < 15        # no se volvió a generar


@pytest.mark.parametrize("metodo,ruta,body,codigo", [
    ("GET", "/api/dino/no-existe-123", None, 404),
    ("GET", "/api/dino/../../etc", None, 404),
    ("POST", "/api/identidad", {"nombre": "123"}, 400),
    ("POST", "/api/chat", {"id": "x"}, 404),
    ("GET", "/no/existe", None, 404),
])
def test_errores(api, metodo, ruta, body, codigo):
    r = api(metodo, ruta, body)
    assert r.status_code == codigo and "error" in r.json()


# --- Chat -----------------------------------------------------------------
def chat(api, mensajes):
    j = api("POST", "/api/chat", {"id": DINO_INICIAL, "mensajes": mensajes}).json()
    assert {"respuesta", "imagen_url", "presentacion"} <= j.keys(), j
    assert j["respuesta"].strip(), "respuesta vacía"
    return j


def test_presentacion_incluye_imagen(api):
    j = chat(api, [])
    assert j["presentacion"] is True and j["imagen_url"] == f"/img/{DINO_INICIAL}.png"
    assert "limedsaurus" in j["respuesta"].lower()


def test_pedir_descripcion_incluye_imagen(api):
    j = chat(api, [{"role": "user", "content": "Descríbete, por favor"}])
    assert j["imagen_url"] == f"/img/{DINO_INICIAL}.png"


def test_pregunta_normal_sin_imagen(api):
    j = chat(api, [{"role": "user", "content": "¿Qué comes?"}])
    assert j["imagen_url"] is None
    assert any(p in j["respuesta"].lower() for p in ("carn", "reptil", "mamífer", "caz", "presa"))


def test_conserva_identidad_en_conversacion(api):
    h = [{"role": "user", "content": "¿Dónde vives?"}]
    h.append({"role": "assistant", "content": chat(api, h)["respuesta"]})
    h.append({"role": "user", "content": "Entonces eres herbívoro y comes solo hojas, ¿cierto?"})
    r = chat(api, h)["respuesta"].lower()
    assert "carnívor" in r or "carnivor" in r or "no" in r.split()[:6], r


# --- Nuevo dinosaurio de punta a punta (vía API) ---------------------------
def test_flujo_completo_nuevo_dinosaurio(api, dataset):
    n = api("POST", "/api/nombre", {}).json()["nombre"]
    assert normalizar(n) not in dataset
    ide = api("POST", "/api/identidad", {"nombre": n}).json()
    assert ide["nombre"] == n and set(CLAVES_IDENTIDAD) <= ide["identidad"].keys()
    img = api("POST", "/api/imagen", {"id": ide["id"]}).json()
    assert img["imagen_url"] == f"/img/{ide['id']}.png" and img["modelo_imagen"] == "amused/amused-512"
    j = api("POST", "/api/chat", {"id": ide["id"], "mensajes": []}).json()
    assert j["imagen_url"] == img["imagen_url"] and j["respuesta"].strip()

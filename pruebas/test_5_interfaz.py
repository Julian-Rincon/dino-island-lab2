"""Sección 5 del PDF probada en un navegador real: ficha, imagen, chat y botón Nuevo dinosaurio."""
import json
import re

import boto3
import pytest
from playwright.sync_api import expect, sync_playwright

from conftest import DINO_INICIAL, RAIZ, SITIO

CAPTURAS = str(RAIZ / "pruebas" / "capturas")


def fijar_dino_inicial():
    boto3.client("s3").put_object(Bucket="dino-lab-170100747321", Key="dinos/actual.json",
                                  Body=json.dumps({"id": DINO_INICIAL}).encode())


@pytest.fixture(scope="module")
def pagina():
    fijar_dino_inicial()
    with sync_playwright() as p:
        nav = p.chromium.launch()
        pg = nav.new_page(viewport={"width": 1400, "height": 1000})
        yield pg
        nav.close()
    fijar_dino_inicial()   # el sitio vuelve a abrir con Limedsaurus


def test_muestra_nombre_descripcion_imagen_y_modelo(pagina):
    pagina.goto(SITIO + "/")
    expect(pagina.locator("#nombre")).to_have_text("Limedsaurus", timeout=30_000)
    expect(pagina.locator("#descripcion")).not_to_be_empty()
    expect(pagina.locator("#foto img")).to_be_visible()
    assert pagina.locator("#modelo dt").count() >= 5          # info del modelo de nombres
    expect(pagina.locator("#modelo")).to_be_visible()                # visible sin hacer clic
    assert "GRU 2 capas x 256" in pagina.locator("#modelo").inner_text()


def test_presentacion_en_chat_con_imagen(pagina):
    burbuja = pagina.locator(".msg.dino").first
    expect(burbuja).to_be_visible(timeout=120_000)
    expect(burbuja.locator("img")).to_be_visible()
    assert len(burbuja.inner_text().strip()) > 20
    pagina.screenshot(path=CAPTURAS + "/1_inicio_limedsaurus.png", full_page=True)


def test_preguntar_por_el_chat(pagina):
    pagina.fill("#entrada", "¿Qué comes?")
    pagina.click("#enviar")
    expect(pagina.locator(".msg.yo")).to_have_count(1)
    expect(pagina.locator(".msg.dino")).to_have_count(2, timeout=120_000)
    assert pagina.locator(".msg.dino").nth(1).locator("img").count() == 0
    pagina.click("#sugerencias >> text=Descríbete")
    expect(pagina.locator(".msg.dino")).to_have_count(3, timeout=120_000)
    expect(pagina.locator(".msg.dino").nth(2).locator("img")).to_be_visible()
    pagina.screenshot(path=CAPTURAS + "/2_chat.png", full_page=True)


def test_boton_nuevo_dinosaurio(pagina):
    pagina.click("#nuevo")
    expect(pagina.locator("#nombre")).not_to_have_text("Limedsaurus", timeout=30_000)
    expect(pagina.locator("#nombre")).to_have_text(re.compile(r"^[A-Z][a-z]{3,}$"), timeout=30_000)
    nuevo = pagina.locator("#nombre").inner_text()
    expect(pagina.locator("#foto img")).to_be_visible(timeout=360_000)
    expect(pagina.locator("#titulo-chat")).to_have_text(f"Habla con {nuevo}")
    burbuja = pagina.locator(".msg.dino").first
    expect(burbuja.locator("img")).to_be_visible(timeout=120_000)
    assert pagina.locator("#error").is_hidden()
    pagina.screenshot(path=CAPTURAS + "/3_nuevo_dinosaurio.png", full_page=True)

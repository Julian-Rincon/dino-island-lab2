"""Sección 2 y 7 del PDF: generador de nombres (preprocesamiento, modelo, curvas, muestreo, 10 nombres)."""
import json
import re
import sys

import numpy as np
import pytest
import torch

from conftest import RAIZ, normalizar

RES = RAIZ / "1_generador_nombres" / "resultados"
SRC = (RAIZ / "1_generador_nombres" / "entrenar.py").read_text()


@pytest.fixture(scope="module")
def pre():
    return json.loads((RES / "preprocesamiento.json").read_text())


@pytest.fixture(scope="module")
def resumen():
    return json.loads((RES / "resumen.json").read_text())


# --- Preprocesamiento (orden de la guía) ---------------------------------
def test_normalizacion_un_solo_criterio(dataset):
    assert all(re.fullmatch(r"[a-z]+", n) for n in dataset)
    assert "def normalizar" in SRC


def test_vocabulario_con_tokens_especiales(pre):
    assert pre["vocabulario"][:3] == ["<PAD>", "<BOS>", "<EOS>"]
    assert len(set(pre["vocabulario"])) == pre["V"] == 29


def test_longitud_maxima_T(pre, dataset):
    # T + 1 = <BOS> + nombre más largo + <EOS>
    assert pre["T"] == max(len(n) for n in dataset) + 1


def test_padding_y_desplazamiento_X_Y(pre):
    X, Y = pre["ejemplo"]["X"], pre["ejemplo"]["Y"]
    assert len(X) == len(Y) == pre["T"]
    assert X[0] == "<BOS>" and X[1:] == Y[:-1]          # Y = X desplazado un carácter
    assert "<EOS>" in Y and Y[-1] == "<PAD>"            # secuencia corta completada con <PAD>


def test_pad_excluido_de_la_perdida():
    assert "CrossEntropyLoss(ignore_index=PAD_ID)" in SRC
    # Comprobación numérica: añadir posiciones <PAD> no cambia la pérdida
    crit = torch.nn.CrossEntropyLoss(ignore_index=0)
    logits = torch.randn(5, 29)
    y = torch.tensor([4, 5, 2, 0, 0])
    assert torch.isclose(crit(logits, y), crit(logits[:3], y[:3]))


# --- Modelo y entrenamiento ----------------------------------------------
def test_dos_configuraciones_registradas(resumen):
    cfgs = resumen["configuraciones"]
    assert len(cfgs) == 2
    for c in cfgs.values():
        assert c["celda"] in ("RNN", "LSTM", "GRU")
        for campo in ("capas", "hidden", "epocas", "funcion_perdida", "optimizador"):
            assert c[campo], campo
    assert {c["celda"] for c in cfgs.values()} == {"LSTM", "GRU"}


def test_curvas_train_y_validacion():
    png = RES / "curvas_perdida.png"
    assert png.exists() and png.stat().st_size > 20_000
    log = (RES / "log_entrenamiento.txt").read_text()
    assert log.count("train") >= 12 and log.count("val") >= 12


# --- Muestreo -------------------------------------------------------------
def test_tres_temperaturas_incluida_1(resumen):
    temps = {e["estrategia"] for e in resumen["muestreo"] if e["estrategia"].startswith("T=")}
    assert len(temps) >= 3 and "T=1.0" in temps


def test_top_k_y_top_p(resumen):
    est = " ".join(e["estrategia"] for e in resumen["muestreo"])
    assert "top-k" in est and "top-p" in est


def test_diez_nombres_nuevos(resumen, dataset):
    diez = resumen["diez_nombres"]
    assert len(diez) == 10 and len(set(diez)) == 10
    copias = [n for n in diez if normalizar(n) in dataset]
    assert not copias, f"copias exactas del dataset: {copias}"


def test_nombre_seleccionado(resumen):
    assert resumen["seleccionado"] in resumen["diez_nombres"]


# --- Modelo exportado para la app (no se reentrena) ------------------------
def test_inferencia_numpy_igual_a_pytorch():
    sys.path.insert(0, str(RAIZ / "4_app" / "lambda"))
    from dino_rnn import GeneradorNombres
    g = GeneradorNombres(RES)
    ck = torch.load(RES / "modelo.pt")
    c, sd = ck["config"], ck["state_dict"]
    emb = torch.nn.Embedding(len(ck["itos"]), c["emb"])
    rnn = getattr(torch.nn, c["celda"])(c["emb"], c["hidden"], c["capas"], batch_first=True)
    out = torch.nn.Linear(c["hidden"], len(ck["itos"]))
    emb.load_state_dict({"weight": sd["emb.weight"]})
    rnn.load_state_dict({k[4:]: v for k, v in sd.items() if k.startswith("rnn.")})
    out.load_state_dict({"weight": sd["out.weight"], "bias": sd["out.bias"]})
    seq = [1] + [g.itos.index(ch) for ch in "velociraptor"]
    with torch.no_grad():
        ref = out(rnn(emb(torch.tensor([seq])))[0])[0].numpy()
    est, mios = g._estado0(), []
    for t in seq:
        l, est = g._paso(t, est)
        mios.append(l)
    assert np.abs(np.array(mios) - ref).max() < 1e-4

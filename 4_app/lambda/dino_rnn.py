"""
Inferencia del modelo de caracteres (GRU/LSTM de PyTorch) solo con numpy.

Carga los pesos exportados por 1_generador_nombres/entrenar.py (modelo_npz.npz +
modelo_meta.json) y genera nombres de forma autoregresiva con temperatura,
top-k y top-p. No reentrena nada: solo lee los pesos ya entrenados.
"""
import json
from pathlib import Path

import numpy as np


def _sig(x):
    return 1.0 / (1.0 + np.exp(-x))


class GeneradorNombres:
    def __init__(self, carpeta):
        carpeta = Path(carpeta)
        self.meta = json.loads((carpeta / "modelo_meta.json").read_text())
        self.w = dict(np.load(carpeta / "modelo_npz.npz"))
        self.itos = self.meta["itos"]
        self.celda = self.meta["celda"]
        self.capas = self.meta["capas"]
        self.H = self.meta["hidden"]
        self.PAD, self.BOS, self.EOS = 0, 1, 2
        ruta = carpeta / "nombres_entrenamiento.txt"
        self.conocidos = set(ruta.read_text().split()) if ruta.exists() else set()

    def _paso(self, token, estado):
        x = self.w["emb.weight"][token]
        nuevo = []
        for l in range(self.capas):
            Wi, Wh = self.w[f"rnn.weight_ih_l{l}"], self.w[f"rnn.weight_hh_l{l}"]
            bi, bh = self.w[f"rnn.bias_ih_l{l}"], self.w[f"rnn.bias_hh_l{l}"]
            if self.celda == "GRU":
                h = estado[l]
                gi, gh = Wi @ x + bi, Wh @ h + bh
                r = _sig(gi[:self.H] + gh[:self.H])
                z = _sig(gi[self.H:2 * self.H] + gh[self.H:2 * self.H])
                n = np.tanh(gi[2 * self.H:] + r * gh[2 * self.H:])
                h = (1 - z) * n + z * h
                nuevo.append(h)
            elif self.celda == "LSTM":
                h, c = estado[l]
                g = Wi @ x + bi + Wh @ h + bh
                i, f = _sig(g[:self.H]), _sig(g[self.H:2 * self.H])
                gg, o = np.tanh(g[2 * self.H:3 * self.H]), _sig(g[3 * self.H:])
                c = f * c + i * gg
                h = o * np.tanh(c)
                nuevo.append((h, c))
            else:  # RNN simple (tanh)
                h = np.tanh(Wi @ x + bi + Wh @ estado[l] + bh)
                nuevo.append(h)
            x = h
        logits = self.w["out.weight"] @ x + self.w["out.bias"]
        return logits, nuevo

    def _estado0(self):
        z = np.zeros(self.H, dtype=np.float32)
        return [(z, z) if self.celda == "LSTM" else z for _ in range(self.capas)]

    def _muestrear(self, logits, rng, temperatura, top_k, top_p):
        p = np.exp((logits - logits.max()) / temperatura)
        p[[self.PAD, self.BOS]] = 0.0
        p /= p.sum()
        if top_k:
            p[p < np.sort(p)[-top_k]] = 0.0
        if top_p:
            orden = np.argsort(-p)
            acum = np.cumsum(p[orden]) / p.sum()
            quitar = orden[(acum - p[orden] / p.sum()) >= top_p]
            p[quitar] = 0.0
        p /= p.sum()
        return int(rng.choice(len(p), p=p))

    def generar(self, temperatura=1.0, top_k=None, top_p=0.9, seed=None, max_len=30):
        rng = np.random.default_rng(seed)
        estado, token, out = self._estado0(), self.BOS, []
        for _ in range(max_len):
            logits, estado = self._paso(token, estado)
            token = self._muestrear(logits, rng, temperatura, top_k, top_p)
            if token == self.EOS:
                break
            out.append(self.itos[token])
        return "".join(out)

    def nombre_nuevo(self, seed=None, **kw):
        """Nombre que no está en el dataset de entrenamiento (5-16 letras)."""
        rng = np.random.default_rng(seed)
        for _ in range(50):
            s = self.generar(seed=int(rng.integers(1 << 31)), **kw)
            if 5 <= len(s) <= 16 and s not in self.conocidos:
                return s.capitalize()
        return s.capitalize()

    def info(self):
        m = self.meta
        return {"arquitectura": f"{m['celda']} {m['capas']} capas x {m['hidden']} unidades",
                "configuracion": m["configuracion"], "parametros": m["parametros"],
                "val_loss": m["mejor_val_loss"], "mejor_epoca": m["mejor_epoca"],
                "dataset": f"{m['dataset']} nombres reales", "vocabulario": len(m["itos"]),
                "muestreo": "top-p = 0.9, temperatura = 1.0"}

"""
Generador de nombres de dinosaurios — modelo de lenguaje a nivel de caracteres.

Etapas (en el orden que pide la guía):
  1. Preprocesamiento: normalizar, vocabulario, <BOS>/<EOS>, T, <PAD>, X/Y desplazados.
  2. Entrenamiento de dos configuraciones (LSTM vs GRU) con pérdida que ignora <PAD>.
  3. Curvas de pérdida train/val de ambas configuraciones.
  4. Muestreo: temperatura (0.5, 1.0, 1.5), top-k y top-p.
  5. Diez nombres nuevos (no presentes en el dataset) + selección de uno.
  6. Exporta el mejor modelo a .npz para inferencia sin PyTorch (Lambda de la app web).

Uso:  python entrenar.py          (CPU, ~2-3 min)
Salidas en ./resultados/
"""
import json
import random
import re
import time
import unicodedata
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

AQUI = Path(__file__).parent
OUT = AQUI / "resultados"
OUT.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Preprocesamiento
# ---------------------------------------------------------------------------
PAD, BOS, EOS = "<PAD>", "<BOS>", "<EOS>"


def normalizar(nombre: str) -> str:
    """Minúsculas, sin tildes y solo letras a-z (mismo criterio para todo)."""
    s = unicodedata.normalize("NFKD", str(nombre)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z]", "", s.lower().strip())


nombres = [normalizar(n) for n in pd.read_csv(AQUI / "data" / "dinos.csv", header=None)[0]]
nombres = sorted(set(n for n in nombres if n))
print(f"Nombres únicos tras normalizar: {len(nombres)}")

caracteres = sorted(set("".join(nombres)))
itos = [PAD, BOS, EOS] + caracteres          # id -> token
stoi = {t: i for i, t in enumerate(itos)}    # token -> id
PAD_ID, BOS_ID, EOS_ID = stoi[PAD], stoi[BOS], stoi[EOS]
V = len(itos)

# Secuencia completa = <BOS> + caracteres + <EOS>;  T + 1 = longitud máxima
secuencias = [[BOS_ID] + [stoi[c] for c in n] + [EOS_ID] for n in nombres]
T = max(len(s) for s in secuencias) - 1      # X e Y tienen longitud T
print(f"Vocabulario: {V} tokens | T = {T}")


def a_tensores(seqs):
    rellenas = [s + [PAD_ID] * (T + 1 - len(s)) for s in seqs]
    data = torch.tensor(rellenas, dtype=torch.long)
    return data[:, :-1], data[:, 1:]          # X = x0..x_{T-1},  Y = x1..x_T


idx = list(range(len(secuencias)))
random.shuffle(idx)
n_val = int(0.1 * len(idx))
val_seqs = [secuencias[i] for i in idx[:n_val]]
train_seqs = [secuencias[i] for i in idx[n_val:]]
Xtr, Ytr = a_tensores(train_seqs)
Xva, Yva = a_tensores(val_seqs)
print(f"Train: {len(Xtr)}  Val: {len(Xva)}")

ejemplo = secuencias[0]
with open(OUT / "preprocesamiento.json", "w") as f:
    json.dump({
        "nombres_unicos": len(nombres), "vocabulario": itos, "V": V, "T": T,
        "train": len(Xtr), "val": len(Xva),
        "ejemplo": {"nombre": nombres[0],
                    "X": [itos[i] for i in (ejemplo + [PAD_ID] * (T + 1 - len(ejemplo)))[:-1]],
                    "Y": [itos[i] for i in (ejemplo + [PAD_ID] * (T + 1 - len(ejemplo)))[1:]]},
    }, f, ensure_ascii=False, indent=1)


# ---------------------------------------------------------------------------
# 2. Modelo y entrenamiento
# ---------------------------------------------------------------------------
class CharRNN(nn.Module):
    def __init__(self, celda, vocab, emb, hidden, capas, dropout):
        super().__init__()
        self.celda = celda
        self.emb = nn.Embedding(vocab, emb, padding_idx=PAD_ID)
        rnn_cls = {"RNN": nn.RNN, "LSTM": nn.LSTM, "GRU": nn.GRU}[celda]
        self.rnn = rnn_cls(emb, hidden, num_layers=capas, batch_first=True,
                           dropout=dropout if capas > 1 else 0.0)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(hidden, vocab)

    def forward(self, x, h=None):
        y, h = self.rnn(self.emb(x), h)
        return self.out(self.drop(y)), h


CONFIGS = [
    {"nombre": "A_LSTM_1x128", "celda": "LSTM", "emb": 32, "hidden": 128, "capas": 1,
     "dropout": 0.0, "epocas": 60, "lr": 3e-3, "batch": 64},
    {"nombre": "B_GRU_2x256", "celda": "GRU", "emb": 48, "hidden": 256, "capas": 2,
     "dropout": 0.3, "epocas": 60, "lr": 2e-3, "batch": 64},
]

criterio = nn.CrossEntropyLoss(ignore_index=PAD_ID)   # <PAD> fuera de la pérdida


def perdida(model, X, Y):
    logits, _ = model(X)
    return criterio(logits.reshape(-1, V), Y.reshape(-1))


resultados = {}
for cfg in CONFIGS:
    torch.manual_seed(SEED)
    model = CharRNN(cfg["celda"], V, cfg["emb"], cfg["hidden"], cfg["capas"], cfg["dropout"])
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    hist = {"train": [], "val": []}
    mejor, mejor_estado, mejor_epoca = float("inf"), None, 0
    t0 = time.time()
    for ep in range(1, cfg["epocas"] + 1):
        model.train()
        perm = torch.randperm(len(Xtr))
        total, n = 0.0, 0
        for i in range(0, len(Xtr), cfg["batch"]):
            b = perm[i:i + cfg["batch"]]
            loss = perdida(model, Xtr[b], Ytr[b])
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total += loss.item() * len(b)
            n += len(b)
        model.eval()
        with torch.no_grad():
            vl = perdida(model, Xva, Yva).item()
        hist["train"].append(total / n)
        hist["val"].append(vl)
        if vl < mejor:
            mejor, mejor_epoca = vl, ep
            mejor_estado = {k: v.clone() for k, v in model.state_dict().items()}
        if ep % 10 == 0:
            print(f"[{cfg['nombre']}] época {ep:3d}  train {total / n:.4f}  val {vl:.4f}")
    model.load_state_dict(mejor_estado)
    params = sum(p.numel() for p in model.parameters())
    resultados[cfg["nombre"]] = {
        "config": cfg, "model": model, "hist": hist, "mejor_val": mejor,
        "mejor_epoca": mejor_epoca, "parametros": params, "segundos": time.time() - t0,
    }
    print(f"[{cfg['nombre']}] mejor val {mejor:.4f} en época {mejor_epoca} | {params} parámetros")

# ---------------------------------------------------------------------------
# 3. Curvas de pérdida
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
for ax, (nom, r) in zip(axes, resultados.items()):
    ax.plot(r["hist"]["train"], label="entrenamiento")
    ax.plot(r["hist"]["val"], label="validación")
    ax.axvline(r["mejor_epoca"] - 1, ls="--", c="gray", lw=0.8, label=f"mejor val (ép. {r['mejor_epoca']})")
    ax.set_title(nom)
    ax.set_xlabel("época")
    ax.grid(alpha=0.3)
    ax.legend()
axes[0].set_ylabel("cross-entropy (sin <PAD>)")
plt.tight_layout()
plt.savefig(OUT / "curvas_perdida.png", dpi=130)

ganador = min(resultados, key=lambda k: resultados[k]["mejor_val"])
print(f"Configuración ganadora: {ganador}")
model = resultados[ganador]["model"].eval()


# ---------------------------------------------------------------------------
# 4. Muestreo
# ---------------------------------------------------------------------------
def siguiente(logits, temperatura=1.0, top_k=None, top_p=None, gen=None):
    probs = torch.softmax(logits / temperatura, dim=-1)
    probs[PAD_ID] = 0.0
    probs[BOS_ID] = 0.0
    if top_k:
        corte = torch.topk(probs, top_k).values[-1]
        probs[probs < corte] = 0.0
    if top_p:
        orden, ids = torch.sort(probs, descending=True)
        acumulada = torch.cumsum(orden / orden.sum(), 0)
        # conjunto más pequeño cuya probabilidad acumulada alcanza p
        quitar = acumulada - orden / orden.sum() >= top_p
        probs[ids[quitar]] = 0.0
    probs = probs / probs.sum()
    return torch.multinomial(probs, 1, generator=gen).item()


@torch.no_grad()
def generar(model, gen, max_len=30, **kw):
    x = torch.tensor([[BOS_ID]])
    h, out = None, []
    for _ in range(max_len):
        logits, h = model(x, h)
        t = siguiente(logits[0, -1], gen=gen, **kw)
        if t == EOS_ID:
            break
        out.append(itos[t])
        x = torch.tensor([[t]])
    return "".join(out)


ESTRATEGIAS = [
    {"etiqueta": "T=0.5", "temperatura": 0.5},
    {"etiqueta": "T=1.0", "temperatura": 1.0},
    {"etiqueta": "T=1.5", "temperatura": 1.5},
    {"etiqueta": "top-k=5 (T=1.0)", "temperatura": 1.0, "top_k": 5},
    {"etiqueta": "top-p=0.9 (T=1.0)", "temperatura": 1.0, "top_p": 0.9},
]
conjunto = set(nombres)
gen = torch.Generator().manual_seed(SEED)
N = 200
comparacion = []
muestras = {}
for e in ESTRATEGIAS:
    kw = {k: v for k, v in e.items() if k != "etiqueta"}
    salida = [generar(model, gen, **kw) for _ in range(N)]
    nuevos = [s for s in salida if s and s not in conjunto]
    largos = [len(s) for s in salida if s]
    comparacion.append({
        "estrategia": e["etiqueta"],
        "% nuevos (no en dataset)": round(100 * len(nuevos) / N, 1),
        "% únicos": round(100 * len(set(salida)) / N, 1),
        "largo medio": round(float(np.mean(largos)), 1),
        "% terminan en -saurus/-raptor/-don/-ops": round(
            100 * sum(bool(re.search(r"(saurus|raptor|don|ops)$", s)) for s in salida) / N, 1),
    })
    muestras[e["etiqueta"]] = list(dict.fromkeys(nuevos))[:8]

tabla = pd.DataFrame(comparacion)
print(tabla.to_string(index=False))

# Diez nombres finales: top-p 0.9 (mejor equilibrio, ver análisis en README)
gen = torch.Generator().manual_seed(7)
diez = []
while len(diez) < 10:
    s = generar(model, gen, temperatura=1.0, top_p=0.9)
    if 5 <= len(s) <= 16 and s not in conjunto and s not in diez:
        diez.append(s)
diez = [s.capitalize() for s in diez]
seleccionado = next((s for s in diez if re.search(r"(saurus|raptor|venator|odon|don)$", s.lower())), diez[0])
print("Diez nombres:", diez)
print("Seleccionado:", seleccionado)

# ---------------------------------------------------------------------------
# 5. Guardar evidencias y exportar el modelo
# ---------------------------------------------------------------------------
torch.save({"state_dict": model.state_dict(), "config": resultados[ganador]["config"],
            "itos": itos}, OUT / "modelo.pt")

pesos = {k: v.numpy().astype(np.float32) for k, v in model.state_dict().items()}
np.savez_compressed(OUT / "modelo_npz.npz", **pesos)
meta = {
    "celda": resultados[ganador]["config"]["celda"],
    "capas": resultados[ganador]["config"]["capas"],
    "hidden": resultados[ganador]["config"]["hidden"],
    "itos": itos, "T": T, "configuracion": ganador,
    "mejor_val_loss": round(resultados[ganador]["mejor_val"], 4),
    "mejor_epoca": resultados[ganador]["mejor_epoca"],
    "parametros": resultados[ganador]["parametros"],
    "dataset": len(nombres),
}
(OUT / "modelo_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
(OUT / "nombres_entrenamiento.txt").write_text("\n".join(nombres))

resumen = {
    "configuraciones": {
        k: {**{c: v for c, v in r["config"].items()},
            "funcion_perdida": "CrossEntropyLoss(ignore_index=<PAD>)",
            "optimizador": f"Adam(lr={r['config']['lr']})",
            "parametros": r["parametros"], "mejor_val_loss": round(r["mejor_val"], 4),
            "mejor_epoca": r["mejor_epoca"],
            "train_loss_final": round(r["hist"]["train"][-1], 4),
            "val_loss_final": round(r["hist"]["val"][-1], 4),
            "segundos": round(r["segundos"], 1)}
        for k, r in resultados.items()},
    "ganador": ganador,
    "muestreo": comparacion,
    "muestras_por_estrategia": muestras,
    "diez_nombres": diez,
    "seleccionado": seleccionado,
}
(OUT / "resumen.json").write_text(json.dumps(resumen, ensure_ascii=False, indent=1))
tabla.to_markdown(OUT / "muestreo.md", index=False) if hasattr(tabla, "to_markdown") else None
print("Listo ->", OUT)

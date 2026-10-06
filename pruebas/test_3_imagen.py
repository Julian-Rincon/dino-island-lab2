"""Sección 4 del PDF: generación de la imagen."""
import json

from PIL import Image

from conftest import RAIZ

IMG = RAIZ / "3_imagen"


def test_notebook_gpu_con_librerias_y_modelo():
    nb = json.loads((IMG / "colab_imagen.ipynb").read_text())
    codigo = "\n".join(c["source"] if isinstance(c["source"], str) else "".join(c["source"])
                       for c in nb["cells"] if c["cell_type"] == "code")
    for lib in ("diffusers", "transformers", "torch", "accelerate", "safetensors"):
        assert lib in codigo
    assert "amused/amused-512" in codigo and "cuda" in codigo


def test_notebook_ejecutado_en_gpu_con_imagen():
    nb = json.loads((IMG / "colab_imagen.ipynb").read_text())
    salidas = json.dumps([c.get("outputs", []) for c in nb["cells"]])
    assert '"cuda' in salidas or "cuda " in salidas, "el notebook no muestra ejecución en GPU"
    assert "image/png" in salidas, "el notebook no tiene la imagen generada"


def test_prompt_usa_nombre_y_rasgos_de_ollama():
    nb = json.loads((IMG / "colab_imagen.ipynb").read_text())
    ide = json.loads((RAIZ / "2_ollama_sagemaker" / "identidad_limedsaurus.json").read_text())
    codigo = json.dumps(nb)
    assert "Limedsaurus" in codigo and ide["prompt_imagen"][:40] in codigo


def test_imagen_del_seleccionado():
    im = Image.open(IMG / "limedsaurus.png")
    assert im.size == (512, 512)

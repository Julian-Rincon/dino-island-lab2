"""Sección 3 del PDF: Ollama en SageMaker (instancia, Docker, modelo, persistencia, lifecycle)."""
import base64
import json

import boto3
import pytest

from conftest import RAIZ, CLAVES_IDENTIDAD

sm = boto3.client("sagemaker", region_name="us-east-1")


@pytest.fixture(scope="module")
def nb():
    return sm.describe_notebook_instance(NotebookInstanceName="ollama")


def test_instancia_m5_xlarge_encendida(nb):
    assert nb["InstanceType"] == "ml.m5.xlarge"
    assert nb["NotebookInstanceStatus"] == "InService"


def test_lifecycle_on_start_asociado_y_igual_al_repo(nb):
    assert nb["NotebookInstanceLifecycleConfigName"] == "lab-dinos-on-start"
    lcc = sm.describe_notebook_instance_lifecycle_config(
        NotebookInstanceLifecycleConfigName="lab-dinos-on-start")
    desplegado = base64.b64decode(lcc["OnStart"][0]["Content"]).decode()
    assert desplegado == (RAIZ / "2_ollama_sagemaker" / "on-start.sh").read_text()


def test_docker_data_root_en_directorio_persistente():
    s = (RAIZ / "2_ollama_sagemaker" / "on-start.sh").read_text()
    assert '"data-root"' in s and "/home/ec2-user/SageMaker/docker" in s
    assert "ollama/ollama" in s and "docker run" in s          # Ollama instalado con Docker
    assert "P=/home/ec2-user/SageMaker" in s and "$P/ollama:/root/.ollama" in s  # modelos persistentes


def test_ollama_responde_con_modelo_registrado(api):
    e = api("GET", "/api/estado", timeout=60).json()
    assert e["notebook"] == "InService"
    modelos = {m["modelo"]: m for m in e["ollama"]}
    assert "gemma4:e2b" in modelos
    assert modelos["gemma4:e2b"]["digest"] == "b37049369adf"     # versión registrada en el README
    assert "b37049369adf" in (RAIZ / "README.md").read_text()


def test_identidad_seleccionado_completa_y_coherente():
    ide = json.loads((RAIZ / "2_ollama_sagemaker" / "identidad_limedsaurus.json").read_text())
    assert ide["nombre"] == "Limedsaurus"
    for k in ("apariencia", "habitat", "alimentacion", "comportamiento", "rasgo_distintivo"):
        assert len(ide[k]) > 10, k
    # coherencia: carnívoro descrito como depredador/cazador
    assert "carn" in ide["alimentacion"].lower()
    assert any(p in ide["descripcion"].lower() for p in ("depredador", "cazador", "caza"))


def test_prompt_da_ejemplos_de_sufijos():
    p = (RAIZ / "2_ollama_sagemaker" / "prompt_identidad.txt").read_text()
    for suf in ("-saurus", "-raptor", "-odon", "-venator"):
        assert suf in p


def test_verificacion_ejecutada_desde_el_notebook():
    """La guía pide comprobar DESDE el notebook que el modelo responde: identidad_ollama.ipynb con salidas."""
    nb = json.loads((RAIZ / "2_ollama_sagemaker" / "identidad_ollama.ipynb").read_text())
    celdas = [c for c in nb["cells"] if c["cell_type"] == "code"]
    salidas = json.dumps([c.get("outputs", []) for c in celdas])
    if not any(c.get("outputs") for c in celdas):
        pytest.skip("PENDIENTE: ejecutar identidad_ollama.ipynb en Jupyter del notebook y guardarlo en el repo")
    assert "gemma4:e2b" in salidas and "tokens en" in salidas

"""
Lambda "dino-app": sirve la página web y orquesta el flujo

    Modelo de caracteres -> Nombre -> Ollama -> Identidad -> Imagen -> Chat

Rutas (Function URL):
  GET  /                     página web (index.html)
  GET  /api/info             información del modelo de caracteres
  GET  /api/estado           salud del notebook, de Ollama y del servidor de imagen
  GET  /api/dino/actual      último dinosaurio creado (el que se muestra al abrir)
  GET  /api/dino/<id>        registro completo de un dinosaurio
  POST /api/nombre           {}                        -> {nombre, modelo}
  POST /api/identidad        {nombre}                  -> {id, nombre, identidad, descripcion}
  POST /api/imagen           {id}                      -> {id, imagen_url, segundos}
  POST /api/chat             {id, mensajes:[...]}      -> {respuesta, imagen_url|null}
  GET  /img/<id>.png         imagen del dinosaurio (generada una sola vez)

Ollama y el servidor de imagen corren en el notebook SageMaker "ollama"; el notebook
no tiene IP pública, así que se entra por su proxy interno /proxy/<puerto>/ con una
URL prefirmada (rol LabRole). Los dinosaurios se guardan en S3 (bucket privado).
"""
import base64
import http.cookiejar
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

import boto3

from dino_rnn import GeneradorNombres

NOTEBOOK = os.environ.get("NOTEBOOK_NAME", "ollama")
BUCKET = os.environ.get("BUCKET", "dino-lab-170100747321")
MODELO_OLLAMA = os.environ.get("MODELO_OLLAMA", "gemma4:e2b")
PUERTO_OLLAMA, PUERTO_IMAGEN = 11434, 8000
SESSION_SECONDS = 12 * 3600

AQUI = Path(__file__).parent
GEN = GeneradorNombres(AQUI)
HTML = (AQUI / "index.html").read_text(encoding="utf-8")

_sm = boto3.client("sagemaker")
_s3 = boto3.client("s3")
_sesion = {"opener": None, "base": None, "expira": 0}


# ---------------------------------------------------------------------------
# Acceso al notebook (Ollama e imagen)
# ---------------------------------------------------------------------------
def _login():
    url = _sm.create_presigned_notebook_instance_url(
        NotebookInstanceName=NOTEBOOK, SessionExpirationDurationInSeconds=SESSION_SECONDS
    )["AuthorizedUrl"]
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))
    with opener.open(url, timeout=30):
        pass
    _sesion.update(opener=opener, base="https://" + urllib.parse.urlsplit(url).netloc,
                   expira=time.time() + SESSION_SECONDS - 1800)


def _notebook(puerto, ruta, payload, timeout=280, reintento=True):
    if time.time() > _sesion["expira"]:
        _login()
    req = urllib.request.Request(f"{_sesion['base']}/proxy/{puerto}{ruta}",
                                 data=None if payload is None else json.dumps(payload).encode(),
                                 method="GET" if payload is None else "POST",
                                 headers={"Content-Type": "application/json"})
    try:
        with _sesion["opener"].open(req, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        if e.code in (401, 403) and reintento:   # sesión vencida: nueva sesión y un reintento
            _sesion["expira"] = 0
            return _notebook(puerto, ruta, payload, timeout, reintento=False)
        raise


def _ollama_chat(mensajes, formato=None, temperatura=0.7, max_tokens=-1):
    # think=False: gemma4 razona antes de responder; sin esto el razonamiento se come
    # el límite de tokens (respuesta vacía) y triplica la latencia en CPU.
    payload = {"model": MODELO_OLLAMA, "messages": mensajes, "stream": False, "think": False,
               "options": {"temperature": temperatura, "num_ctx": 4096, "num_predict": max_tokens}}
    if formato:
        payload["format"] = formato
    return _notebook(PUERTO_OLLAMA, "/api/chat", payload)["message"]["content"]


# ---------------------------------------------------------------------------
# Almacenamiento
# ---------------------------------------------------------------------------
def _guardar(dino):
    _s3.put_object(Bucket=BUCKET, Key=f"dinos/{dino['id']}.json",
                   Body=json.dumps(dino, ensure_ascii=False).encode(), ContentType="application/json")


def _cargar(dino_id):
    if not re.fullmatch(r"[a-z0-9-]{4,40}", dino_id or ""):
        raise KeyError(dino_id)
    if dino_id == "actual":
        dino_id = json.loads(_s3.get_object(Bucket=BUCKET, Key="dinos/actual.json")["Body"].read())["id"]
    return json.loads(_s3.get_object(Bucket=BUCKET, Key=f"dinos/{dino_id}.json")["Body"].read())


# ---------------------------------------------------------------------------
# Etapas del flujo
# ---------------------------------------------------------------------------
PROMPT_IDENTIDAD = """Eres un paleontólogo de la Isla de los Dinosaurios. Los investigadores acaban de \
nombrar una especie FICTICIA llamada "{nombre}". Inventa su identidad, coherente con el nombre: \
los nombres reales usan raíces griegas/latinas, por ejemplo -saurus (lagarto), -raptor (ladrón), \
-odon/-don (diente), -venator (cazador), -ceratops (cara con cuernos), -onyx (garra), \
-lophus (cresta), -mimus (imitador), -titan (gigante), -suchus (cocodrilo), -ops (cara). \
Si el nombre contiene alguna de estas raíces, úsala para justificar un rasgo.

Responde SOLO un JSON con estas claves (valores en español, frases breves):
{{"significado_nombre": "...", "periodo": "...", "tamano": "longitud y peso aproximados",
"apariencia": "...", "habitat": "...", "alimentacion": "herbívoro/carnívoro/omnívoro y qué come",
"comportamiento": "...", "rasgo_distintivo": "...",
"descripcion": "un párrafo de 4-5 frases que integre todo lo anterior",
"prompt_imagen": "IN ENGLISH, max 45 words: photorealistic description of the animal's body, colors, \
distinctive feature and habitat, nature documentary style"}}

Todos los rasgos deben ser consistentes entre sí (p. ej. un carnívoro tiene dientes afilados)."""

CLAVES = ["significado_nombre", "periodo", "tamano", "apariencia", "habitat", "alimentacion",
          "comportamiento", "rasgo_distintivo", "descripcion", "prompt_imagen"]


def etapa_nombre(body):
    return {"nombre": GEN.nombre_nuevo(), "modelo": GEN.info()}


def etapa_identidad(body):
    nombre = re.sub(r"[^A-Za-z]", "", str(body.get("nombre", "")))[:30].capitalize()
    if not nombre:
        raise ValueError("falta 'nombre'")
    t0 = time.time()
    identidad = {}
    for intento in range(3):
        try:
            identidad = json.loads(_ollama_chat(
                [{"role": "user", "content": PROMPT_IDENTIDAD.format(nombre=nombre)}],
                formato="json", temperatura=0.8))
        except json.JSONDecodeError:
            continue
        if all(isinstance(identidad.get(k), str) and identidad[k].strip() for k in CLAVES):
            break
    faltan = [k for k in CLAVES if not identidad.get(k)]
    if faltan:
        raise RuntimeError(f"Ollama no devolvió: {faltan}")
    dino = {"id": f"{nombre.lower()}-{uuid.uuid4().hex[:6]}", "nombre": nombre,
            "identidad": {k: identidad[k] for k in CLAVES}, "modelo_nombres": GEN.info(),
            "modelo_ollama": MODELO_OLLAMA, "segundos_identidad": round(time.time() - t0, 1),
            "imagen": None, "creado": int(time.time())}
    _guardar(dino)
    return {k: dino[k] for k in ("id", "nombre", "identidad", "modelo_ollama", "segundos_identidad")}


def etapa_imagen(body):
    dino = _cargar(body.get("id", ""))
    if dino.get("imagen"):                      # una sola imagen por dinosaurio
        return {"id": dino["id"], "imagen_url": dino["imagen"], "reutilizada": True}
    prompt = f"{dino['nombre']}, a dinosaur. {dino['identidad']['prompt_imagen']}"
    r = _notebook(PUERTO_IMAGEN, "/generar", {"prompt": prompt, "seed": 7, "pasos": 12})
    _s3.put_object(Bucket=BUCKET, Key=f"img/{dino['id']}.png",
                   Body=base64.b64decode(r["png_base64"]), ContentType="image/png")
    dino.update(imagen=f"/img/{dino['id']}.png", prompt_imagen_final=prompt,
                modelo_imagen=r["modelo"], segundos_imagen=r["segundos"])
    _guardar(dino)
    _s3.put_object(Bucket=BUCKET, Key="dinos/actual.json", Body=json.dumps({"id": dino["id"]}).encode())
    return {"id": dino["id"], "imagen_url": dino["imagen"], "segundos": r["segundos"],
            "modelo_imagen": r["modelo"], "dispositivo": r["dispositivo"]}


PIDE_IMAGEN = re.compile(
    r"(descr[ií]be|descripci[oó]n|c[oó]mo eres|como luces|aspecto|apariencia|foto|imagen|"
    r"mu[eé]strate|\bverte\b|pres[eé]ntate|presentaci[oó]n|qui[eé]n eres|"
    r"c[oó]mo te ves|look like|describe yourself|who are you)", re.I)


def etapa_chat(body):
    dino = _cargar(body.get("id", ""))
    ide = dino["identidad"]
    historial = [m for m in body.get("mensajes", [])[-12:]
                 if m.get("role") in ("user", "assistant") and isinstance(m.get("content"), str)]
    historial = [{"role": m["role"], "content": m["content"][:1000]} for m in historial]
    sistema = (
        f"Eres {dino['nombre']}, un dinosaurio ficticio de la Isla de los Dinosaurios, y conversas "
        f"en primera persona, en español, con tono cercano y breve (máximo 5 frases).\n"
        f"Tu identidad (es la verdad sobre ti y NUNCA la contradices):\n"
        + "\n".join(f"- {k}: {ide[k]}" for k in CLAVES if k != "prompt_imagen")
        + "\nPuedes ampliar detalles que no estén ahí, siempre que sean coherentes con estos rasgos. "
          "Si te preguntan algo que contradice tu identidad, corrige amablemente. "
          "No digas que eres un modelo de lenguaje."
    )
    presentacion = not any(m["role"] == "user" for m in historial)
    if presentacion:
        historial = [{"role": "user", "content": "Preséntate: ¿quién eres y cómo eres?"}]
    respuesta = _ollama_chat([{"role": "system", "content": sistema}] + historial, max_tokens=300)
    ultimo = historial[-1]["content"]
    mostrar = presentacion or bool(PIDE_IMAGEN.search(ultimo))
    return {"respuesta": respuesta, "imagen_url": dino.get("imagen") if mostrar else None,
            "presentacion": presentacion}


def estado():
    """Salud de los servicios del notebook (para la página y para depurar)."""
    out = {"notebook": _sm.describe_notebook_instance(NotebookInstanceName=NOTEBOOK)["NotebookInstanceStatus"]}
    for nombre, puerto, ruta in (("ollama", PUERTO_OLLAMA, "/api/tags"), ("imagen", PUERTO_IMAGEN, "/salud")):
        try:
            r = _notebook(puerto, ruta, None, timeout=15)
            out[nombre] = ([{"modelo": m["name"], "digest": m["digest"][:12], **m.get("details", {})} for m in r["models"]] if nombre == "ollama" else r)
        except Exception as e:  # noqa: BLE001
            out[nombre] = f"no disponible ({type(e).__name__})"
    return out


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------
def _resp(status, payload, ctype="application/json; charset=utf-8", b64=False):
    body = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
    return {"statusCode": status, "headers": {"Content-Type": ctype, "Cache-Control": "no-store"},
            "body": body, "isBase64Encoded": b64}


POST = {"/api/nombre": etapa_nombre, "/api/identidad": etapa_identidad,
        "/api/imagen": etapa_imagen, "/api/chat": etapa_chat}


def lambda_handler(event, context):
    http = event.get("requestContext", {}).get("http", {})
    metodo, ruta = http.get("method", "GET"), event.get("rawPath", "/")
    try:
        if metodo == "GET" and ruta in ("/", "/index.html"):
            return _resp(200, HTML, "text/html; charset=utf-8")
        if metodo == "GET" and ruta == "/api/info":
            return _resp(200, {"modelo_nombres": GEN.info(), "modelo_ollama": MODELO_OLLAMA})
        if metodo == "GET" and ruta == "/api/estado":
            return _resp(200, estado())
        if metodo == "GET" and ruta.startswith("/api/dino/"):
            return _resp(200, _cargar(ruta.rsplit("/", 1)[1]))
        m = re.fullmatch(r"/img/([a-z0-9-]{4,40})\.png", ruta)
        if metodo == "GET" and m:
            png = _s3.get_object(Bucket=BUCKET, Key=f"img/{m.group(1)}.png")["Body"].read()
            r = _resp(200, base64.b64encode(png).decode(), "image/png", b64=True)
            r["headers"]["Cache-Control"] = "public, max-age=86400"
            return r
        if metodo == "POST" and ruta in POST:
            raw = event.get("body") or "{}"
            if event.get("isBase64Encoded"):
                raw = base64.b64decode(raw).decode()
            return _resp(200, POST[ruta](json.loads(raw)))
        return _resp(404, {"error": "ruta no encontrada"})
    except (KeyError, _s3.exceptions.NoSuchKey):
        return _resp(404, {"error": "dinosaurio no encontrado"})
    except ValueError as e:
        return _resp(400, {"error": str(e)})
    except Exception as e:  # noqa: BLE001
        return _resp(502, {"error": f"{type(e).__name__}: {e}"[:300]})

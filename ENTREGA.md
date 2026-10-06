# Laboratorio II — Modelos de lenguaje a nivel de caracteres con RNN

**Procesamiento de Lenguaje Natural — 2026 S02 — Universidad Sergio Arboleda**<br>
Estudiante: Julián Rincón

## Entregables

| Entregable | Enlace / ubicación |
|---|---|
| Sitio web funcionando | https://zk6dst5eom6dhbjfcivxejcpn40iakbw.lambda-url.us-east-1.on.aws/ |
| Repositorio | https://github.com/Julian-Rincon/dino-island-lab2 |
| README con instrucciones | `README.md` en la raíz del repositorio |
| Generador de nombres | `1_generador_nombres/entrenar.py` |
| Configuración de Ollama | `2_ollama_sagemaker/on-start.sh`, `identidad_ollama.ipynb`, `prompt_identidad.txt` |
| Código de generación de imágenes | `3_imagen/colab_imagen.ipynb`, `3_imagen/servidor_imagen.py` |
| Archivos de la aplicación | `4_app/lambda/` (página + orquestador) y `4_app/deploy.sh` |
| Evidencias experimentales | `1_generador_nombres/resultados/` (configuración, curvas, muestreo, 10 nombres) |

> El sitio depende del notebook de SageMaker `ollama`; debe estar encendido para usarlo.

## Cómo se realizó

**1. Nombre.** Se normalizaron los 1536 nombres de `dinos.csv` (minúsculas, solo a-z), se construyó
el vocabulario con `<PAD>`, `<BOS>` y `<EOS>` (29 tokens), se fijó T = 27, se rellenó con `<PAD>` y se
formaron X e Y desplazadas un carácter. Se compararon dos modelos: **LSTM 1×128** y **GRU 2×256**,
con Adam y entropía cruzada que ignora `<PAD>`. Las curvas muestran que ambos sobreajustan después
de la época 11–14, así que se guardó la mejor época de validación. Ganó la GRU (pérdida de validación 1,564).
Se compararon temperaturas 0,5, 1,0 y 1,5, top-k = 5 y top-p = 0,9. Top-p 0,9 dio el mejor
equilibrio entre nombres nuevos y forma de nombre real. De los 10 nombres nuevos se eligió **Limedsaurus**.

**2. Identidad.** En un notebook de SageMaker `ml.m5.xlarge` se instaló Ollama con Docker y se
descargó `gemma4:e2b` (digest `b37049369adf`, 4.6B, Q4_K_M). Un lifecycle script *on-start* deja el
data-root de Docker, los modelos y la caché en `/home/ec2-user/SageMaker`, así que todo sobrevive a
los reinicios. El prompt da ejemplos de raíces reales (-saurus, -raptor, -odon, -venator…) y pide en
JSON la apariencia, el hábitat, la alimentación, el comportamiento y el rasgo distintivo. Resultado: Limedsaurus es un
depredador del Jurásico Tardío, gris pizarra, con placas óseas y una cresta para intimidar.

**3. Imagen.** Con `amused/amused-512` (diffusers, transformers, torch, accelerate, safetensors)
se generó la imagen a partir del nombre y los rasgos de Ollama. El notebook se ejecutó en GPU
(1,6 s por imagen). La app usa el mismo código como servicio dentro del notebook de SageMaker (CPU,
~45 s). Así no depende de una sesión de Colab abierta.

**4. Integración.** Una función AWS Lambda con URL pública sirve la página y orquesta el flujo
*nombre → identidad → imagen → chat*. Hace el muestreo con la GRU exportada a numpy, sin
reentrenar. Llega a Ollama y al generador de imágenes por el proxy interno del notebook, con una URL
prefirmada. Los dinosaurios y sus imágenes se guardan en un bucket S3 privado. Cada imagen se genera una sola vez y se reutiliza. El
chat envía la identidad como contexto en cada mensaje y el dinosaurio no la contradice. Al
presentarse o al describirse, el mensaje incluye la imagen. El botón **Nuevo dinosaurio** repite todo
en unos 2 minutos. El README documenta qué recibe y qué devuelve cada componente.

## Verificación (sección 9 de la guía)

Se automatizó en `pruebas/` (pytest y un navegador real con Playwright, contra el sitio desplegado).
El resultado está en `pruebas/REPORTE.md`.

| Punto de la guía | Resultado |
|---|---|
| Los 10 nombres no son copias del dataset | ✔ comprobado contra `dinos.csv` |
| Identidad coherente | ✔ carnívoro ↔ depredador, dientes, caza |
| La imagen corresponde a los rasgos | ✔ gris, placas, pantano brumoso (revisión visual) |
| El chat conserva la identidad | ✔ ante "¿eres herbívoro?" responde que es carnívoro |
| La imagen aparece al presentarse o describirse | ✔ y no aparece en preguntas normales |
| Nuevo Dinosaurio sin reentrenar | ✔ probado con el botón en el navegador |
| Sitio accesible | ✔ |
| Repositorio con instrucciones y sin información sensible | ✔ búsqueda automática de claves y tokens |

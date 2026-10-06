# Servidor de imagen para el notebook de SageMaker (CPU). En Colab se usa colab_imagen.ipynb.
FROM python:3.11-slim
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir "diffusers==0.41.0" "transformers==5.19.0" accelerate safetensors fastapi uvicorn pillow
WORKDIR /app
COPY servidor_imagen.py .
ENV HF_HOME=/hf
EXPOSE 8000
CMD ["uvicorn", "servidor_imagen:app", "--host", "0.0.0.0", "--port", "8000"]

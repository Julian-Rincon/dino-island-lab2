#!/bin/bash
# Lifecycle config "on-start" del notebook SageMaker "ollama" (ml.m5.xlarge).
# Se ejecuta como root en CADA arranque. SageMaker solo conserva
# /home/ec2-user/SageMaker entre reinicios, así que todo vive ahí:
#   - data-root de Docker   -> /home/ec2-user/SageMaker/docker
#   - modelos de Ollama     -> /home/ec2-user/SageMaker/ollama
#   - caché de Hugging Face -> /home/ec2-user/SageMaker/hf
# El script tiene 5 min de límite: las tareas largas (pull del modelo, build
# de la imagen del servidor) se lanzan en segundo plano con nohup.
set -euxo pipefail

P=/home/ec2-user/SageMaker
BUCKET=dino-lab-170100747321
MODELO_OLLAMA=gemma4:e2b
LOG=$P/lab-dinos/arranque.log

mkdir -p "$P/docker" "$P/ollama" "$P/hf" "$P/lab-dinos"

# 1) Docker con data-root persistente (se conserva el resto de daemon.json)
python3 - <<'PY'
import json, os
f = "/etc/docker/daemon.json"
cfg = json.load(open(f)) if os.path.exists(f) and os.path.getsize(f) else {}
cfg["data-root"] = "/home/ec2-user/SageMaker/docker"
json.dump(cfg, open(f, "w"), indent=2)
PY
systemctl restart docker
docker info --format '{{.DockerRootDir}}'

# 2) Código del servidor de imagen (publicado en S3 por deploy.sh)
aws s3 sync "s3://$BUCKET/notebook/" "$P/lab-dinos/" --exact-timestamps || true
chown -R ec2-user:ec2-user "$P/lab-dinos"

# 3) Ollama (solo CPU) con los modelos en el disco persistente
if docker inspect ollama >/dev/null 2>&1; then
  docker start ollama
else
  docker run -d --name ollama --restart unless-stopped \
    -p 11434:11434 -v "$P/ollama:/root/.ollama" \
    -e OLLAMA_KEEP_ALIVE=30m ollama/ollama
fi

# 4) Tareas largas en segundo plano
nohup bash -c "
  set -x
  until curl -sf localhost:11434/api/version; do sleep 2; done
  docker exec ollama ollama pull $MODELO_OLLAMA
  docker exec ollama ollama list

  cd $P/lab-dinos
  docker build -t dino-imagen . && {
    docker rm -f dino-imagen 2>/dev/null
    docker run -d --name dino-imagen --restart unless-stopped \
      -p 8000:8000 -v $P/hf:/hf dino-imagen
  }
  echo ARRANQUE_COMPLETO \$(date)
" >> "$LOG" 2>&1 &

echo "on-start terminado (tareas largas siguen en $LOG)"

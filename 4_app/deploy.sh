#!/usr/bin/env bash
# Empaqueta y despliega la Lambda "dino-app" (página + orquestador) con Function URL.
# Requiere: credenciales AWS del Learner Lab (aws configure) y el modelo exportado
# en ../1_generador_nombres/resultados/.
set -euo pipefail
cd "$(dirname "$0")"
FUNC=dino-app
BUCKET=dino-lab-170100747321
ROLE=$(aws iam get-role --role-name LabRole --query Role.Arn --output text)

rm -rf build function.zip && mkdir build
pip install -q --platform manylinux2014_x86_64 --only-binary=:all: --python-version 3.12 numpy -t build
cp lambda/{lambda_function.py,dino_rnn.py,index.html} build/
cp ../1_generador_nombres/resultados/{modelo_npz.npz,modelo_meta.json,nombres_entrenamiento.txt} build/
(cd build && zip -qr ../function.zip .)

if aws lambda get-function --function-name "$FUNC" >/dev/null 2>&1; then
  aws lambda update-function-code --function-name "$FUNC" --zip-file fileb://function.zip >/dev/null
else
  aws lambda create-function --function-name "$FUNC" --runtime python3.12 \
    --handler lambda_function.lambda_handler --role "$ROLE" --zip-file fileb://function.zip \
    --timeout 300 --memory-size 1024 \
    --environment "Variables={BUCKET=$BUCKET,NOTEBOOK_NAME=ollama,MODELO_OLLAMA=gemma4:e2b}" >/dev/null
  aws lambda wait function-active --function-name "$FUNC"
  aws lambda create-function-url-config --function-name "$FUNC" --auth-type NONE >/dev/null
  aws lambda add-permission --function-name "$FUNC" --statement-id url-publica \
    --action lambda:InvokeFunctionUrl --principal '*' --function-url-auth-type NONE >/dev/null
  # Desde oct-2025 una Function URL pública también necesita lambda:InvokeFunction
  aws lambda add-permission --function-name "$FUNC" --statement-id url-publica-invoke \
    --action lambda:InvokeFunction --principal '*' --invoked-via-function-url >/dev/null
fi
aws lambda wait function-updated --function-name "$FUNC"
echo "Sitio: $(aws lambda get-function-url-config --function-name "$FUNC" --query FunctionUrl --output text)"

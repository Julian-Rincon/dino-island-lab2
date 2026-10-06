# Reporte de pruebas

Corrida: 2026-10-06 contra https://zk6dst5eom6dhbjfcivxejcpn40iakbw.lambda-url.us-east-1.on.aws

**58 de 59 pruebas pasan**, 1 pendiente(s). Duración total: 392 s.


## §2 y §7 Generador de nombres

| Prueba | Resultado | s |
|---|---|---:|
| `test_normalizacion_un_solo_criterio` | ✔ | 0.0 |
| `test_vocabulario_con_tokens_especiales` | ✔ | 0.0 |
| `test_longitud_maxima_T` | ✔ | 0.0 |
| `test_padding_y_desplazamiento_X_Y` | ✔ | 0.0 |
| `test_pad_excluido_de_la_perdida` | ✔ | 0.0 |
| `test_dos_configuraciones_registradas` | ✔ | 0.0 |
| `test_curvas_train_y_validacion` | ✔ | 0.0 |
| `test_tres_temperaturas_incluida_1` | ✔ | 0.0 |
| `test_top_k_y_top_p` | ✔ | 0.0 |
| `test_diez_nombres_nuevos` | ✔ | 0.0 |
| `test_nombre_seleccionado` | ✔ | 0.0 |
| `test_inferencia_numpy_igual_a_pytorch` | ✔ | 0.0 |

## §3 Ollama en SageMaker

| Prueba | Resultado | s |
|---|---|---:|
| `test_instancia_m5_xlarge_encendida` | ✔ | 0.3 |
| `test_lifecycle_on_start_asociado_y_igual_al_repo` | ✔ | 0.1 |
| `test_docker_data_root_en_directorio_persistente` | ✔ | 0.0 |
| `test_ollama_responde_con_modelo_registrado` | ✔ | 0.5 |
| `test_identidad_seleccionado_completa_y_coherente` | ✔ | 0.0 |
| `test_prompt_da_ejemplos_de_sufijos` | ✔ | 0.0 |
| `test_verificacion_ejecutada_desde_el_notebook` | ⏸ pendiente | 0.0 |

## §4 Imagen

| Prueba | Resultado | s |
|---|---|---:|
| `test_notebook_gpu_con_librerias_y_modelo` | ✔ | 0.0 |
| `test_notebook_ejecutado_en_gpu_con_imagen` | ✔ | 0.0 |
| `test_prompt_usa_nombre_y_rasgos_de_ollama` | ✔ | 0.0 |
| `test_imagen_del_seleccionado` | ✔ | 0.0 |

## §5 y §9 Contratos de la API y chat

| Prueba | Resultado | s |
|---|---|---:|
| `test_sitio_accesible` | ✔ | 0.2 |
| `test_contrato_info` | ✔ | 0.1 |
| `test_contrato_nombre_sin_reentrenar` | ✔ | 0.6 |
| `test_contrato_dino_actual` | ✔ | 0.2 |
| `test_contrato_imagen_png` | ✔ | 0.5 |
| `test_imagen_se_genera_una_vez_y_se_reutiliza` | ✔ | 0.2 |
| `test_errores[GET-/api/dino/no-existe-123-None-404]` | ✔ | 0.1 |
| `test_errores[GET-/api/dino/../../etc-None-404]` | ✔ | 0.1 |
| `test_errores[POST-/api/identidad-body2-400]` | ✔ | 0.1 |
| `test_errores[POST-/api/chat-body3-404]` | ✔ | 0.1 |
| `test_errores[GET-/no/existe-None-404]` | ✔ | 0.1 |
| `test_presentacion_incluye_imagen` | ✔ | 12.9 |
| `test_pedir_descripcion_incluye_imagen` | ✔ | 20.9 |
| `test_pregunta_normal_sin_imagen` | ✔ | 19.8 |
| `test_conserva_identidad_en_conversacion` | ✔ | 24.5 |
| `test_flujo_completo_nuevo_dinosaurio` | ✔ | 131.0 |

## §5 Interfaz (navegador real)

| Prueba | Resultado | s |
|---|---|---:|
| `test_muestra_nombre_descripcion_imagen_y_modelo` | ✔ | 1.6 |
| `test_presentacion_en_chat_con_imagen` | ✔ | 24.6 |
| `test_preguntar_por_el_chat` | ✔ | 24.9 |
| `test_boton_nuevo_dinosaurio` | ✔ | 127.4 |

## §6, §7 y §9 Repositorio

| Prueba | Resultado | s |
|---|---|---:|
| `test_entregable_en_repo[1_generador_nombres/entrenar.py]` | ✔ | 0.0 |
| `test_entregable_en_repo[2_ollama_sagemaker/on-start.sh]` | ✔ | 0.0 |
| `test_entregable_en_repo[2_ollama_sagemaker/identidad_ollama.ipynb]` | ✔ | 0.0 |
| `test_entregable_en_repo[3_imagen/colab_imagen.ipynb]` | ✔ | 0.0 |
| `test_entregable_en_repo[3_imagen/servidor_imagen.py]` | ✔ | 0.0 |
| `test_entregable_en_repo[4_app/lambda/lambda_function.py]` | ✔ | 0.0 |
| `test_entregable_en_repo[4_app/lambda/index.html]` | ✔ | 0.0 |
| `test_entregable_en_repo[4_app/deploy.sh]` | ✔ | 0.0 |
| `test_entregable_en_repo[README.md]` | ✔ | 0.0 |
| `test_entregable_en_repo[ENTREGA.md]` | ✔ | 0.0 |
| `test_evidencias_experimentales[1_generador_nombres/resultados/resumen.json]` | ✔ | 0.0 |
| `test_evidencias_experimentales[1_generador_nombres/resultados/curvas_perdida.png]` | ✔ | 0.0 |
| `test_evidencias_experimentales[1_generador_nombres/resultados/muestreo.md]` | ✔ | 0.0 |
| `test_readme_documenta_componentes` | ✔ | 0.0 |
| `test_sin_credenciales_ni_tokens` | ✔ | 0.0 |
| `test_sin_archivos_sensibles` | ✔ | 0.0 |


**Pendiente:** `identidad_ollama.ipynb` debe ejecutarse una vez desde Jupyter del notebook `ollama` (la guía pide comprobar *desde el notebook* que el modelo responde) y guardarse en el repo con sus salidas.

Capturas del navegador: `capturas/1_inicio_limedsaurus.png`, `capturas/2_chat.png`, `capturas/3_nuevo_dinosaurio.png`.

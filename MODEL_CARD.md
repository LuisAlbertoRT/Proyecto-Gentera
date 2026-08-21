# Model Card: Prediccion de egreso por mejoria

## 1. Resumen

Este modelo estima la probabilidad de que un paciente egrese por mejoria. Es un problema de clasificacion binaria:

- `1`: egreso por mejoria.
- `0`: cualquier otro motivo de egreso.

El modelo esta pensado como apoyo analitico para priorizar revision y analizar patrones. No reemplaza la valoracion clinica ni debe utilizarse como unica base para una decision sobre un paciente.

## 2. Modelo

- Algoritmo: `RandomForestClassifier`.
- Preprocesamiento: pipeline de scikit-learn.
- Imputacion numerica: mediana, con indicadores de valores faltantes.
- Escalamiento numerico: `StandardScaler`.
- Variables categoricas: imputacion por moda y `OneHotEncoder`.
- Categorias nuevas: se ignoran mediante `handle_unknown="ignore"`.
- Semilla: `42`.

### Hiperparametros principales

```text
n_estimators=500
max_depth=14
min_samples_leaf=40
max_features="sqrt"
class_weight="balanced"
criterion="gini"
bootstrap=True
```

La profundidad maxima y el numero minimo de observaciones por hoja se eligieron para reducir la varianza y controlar el overfitting. `class_weight="balanced"` compensa el desbalance entre clases durante el entrenamiento.

## 3. Datos de entrenamiento

El flujo procesa un libro Excel con una hoja principal `Base de datos` y catalogos. El entrenamiento registrado contiene:

- Filas procesadas: `29,894`.
- Catalogo de estatus: identifica automaticamente el codigo de mejoria.
- Codigo positivo registrado: `2`.
- Catalogo de enfermedades: agrega descripcion, tipo de problema y nivel de gravedad.
- Catalogo de ciudades: agrega la descripcion de la entidad cuando esta disponible.

Los catalogos se utilizan para interpretar identificadores y enriquecer los datos. Las uniones se validan como muchos-a-uno para evitar duplicar pacientes.

## 4. Variables de entrada

El modelo utiliza estas ocho variables:

- `DIAS_ESTANCIA`: dias de estancia.
- `EDAD`: edad del paciente.
- `GENERO`: genero codificado.
- `NIVEL_DE_GRAVEDAD`: nivel de gravedad obtenido del catalogo.
- `IMC`: indice de masa corporal.
- `ES_INDIGENA`: indicador o codigo de pertenencia indigena.
- `ENTIDAD`: entidad federativa codificada.
- `MES`: mes asociado al registro.

Se evaluaron representaciones alternativas de edad: `EDAD_CUADRADO`, `GRUPO_ETARIO` y sus combinaciones. En la comparacion documentada, `EDAD` fue la representacion seleccionada.

## 5. Variables excluidas

`MOTIVO_EGRESO` no se utiliza como predictor porque define directamente la respuesta. Incluirla produciria fuga de informacion y metricas artificialmente altas.

Tambien se excluyen identificadores administrativos y descripciones que no forman parte del escenario final validado.

## 6. Desempeno registrado

Metricas del entrenamiento automatizado almacenadas en `entregable/Salida/metadata_modelo.json`:

| Metrica | Resultado |
|---|---:|
| ROC-AUC promedio de validacion cruzada | 0.7575 |
| ROC-AUC en prueba | 0.7693 |
| F1 | 0.8202 |
| Precision | 0.9479 |
| Recall | 0.7229 |
| Accuracy | 0.7160 |
| Brier score | 0.1873 |
| Average precision | 0.9633 |

La accuracy no debe interpretarse de forma aislada. La clase positiva es mayoritaria en los datos disponibles, por lo que precision, recall, F1, ROC-AUC y average precision aportan una evaluacion mas informativa.

## 7. Validacion y control de overfitting

El flujo realiza:

- Separacion estratificada de entrenamiento y prueba.
- Validacion cruzada estratificada de cinco particiones.
- Imputacion y transformacion dentro del pipeline para evitar fuga de informacion.
- Comparacion contra regresion logistica, SVM lineal calibrada y baseline.
- Curvas de aprendizaje.
- Comparacion entre ROC-AUC de entrenamiento y validacion.
- Seleccion de hiperparametros regularizados.

La evidencia detallada se encuentra en:

- `Importacion y analisis.ipynb`.
- `Notebooks/Outputs/comparacion_modelos.csv`.
- `Notebooks/Outputs/comparacion_modelos_calibrados.csv`.
- `Notebooks/Outputs/reporte_sesgo_varianza.csv`.
- `Notebooks/Outputs/importancia_permutacion.csv`.
- `Notebooks/Outputs/metadata_modelo.json`.

El resultado debe interpretarse como validacion tecnica sobre esta base de datos, no como validacion clinica externa.

## 8. Uso previsto

Usos apropiados:

- Estimar una probabilidad de egreso por mejoria.
- Priorizar analisis o revision de casos.
- Analizar patrones agregados de egreso.
- Comparar cambios futuros en la calidad o distribucion de los datos.

Usos no apropiados:

- Decidir altas medicas automaticamente.
- Sustituir al personal medico.
- Inferir causalidad entre una variable y el egreso.
- Aplicar el modelo a poblaciones con distribuciones muy distintas sin revalidarlo.
- Usar la prediccion como diagnostico.

## 9. Limitaciones

- Las metricas dependen de la calidad y representatividad del archivo de entrenamiento.
- La evaluacion no constituye validacion externa ni prospectiva.
- El modelo puede degradarse ante cambios en hospitales, poblacion, politicas de egreso o codificacion.
- Las categorias y codigos deben conservar significados compatibles con los catalogos usados.
- Un valor de probabilidad no es una certeza clinica.
- El umbral operativo actual es `0.5`; debe ajustarse segun el costo relativo de falsos positivos y falsos negativos.
- No se ha documentado una auditoria formal de equidad por subgrupos.

## 10. Riesgos y equidad

Variables como entidad, genero, edad e indicador indigena pueden reflejar diferencias estructurales o de registro, no necesariamente diferencias clinicas causales. Antes de usar el modelo en produccion se recomienda revisar precision, recall, calibration y tasas de error por subgrupo.

No deben tomarse decisiones adversas contra un paciente usando exclusivamente estas variables o la prediccion resultante.

## 11. Interpretabilidad

La importancia por permutacion permite estimar cuanto cambia el ROC-AUC cuando se desordena cada variable. Este resultado describe dependencia predictiva del modelo; no demuestra causalidad.

Para explicaciones individuales se recomienda agregar una tecnica como SHAP, revisando previamente sus costos, estabilidad y compatibilidad con el pipeline.

## 12. Artefactos y reproducibilidad

El pipeline entrenado se guarda en:

```text
entregable/Salida/modelo_egreso_mejora.joblib
```

Los metadatos y resultados se guardan en:

```text
entregable/Salida/metadata_modelo.json
Notebooks/Outputs/
```

La API se ejecuta con `5) api_fastapi.py` y expone `/health`, `/predict` y `/docs`. La ejecucion en Docker esta documentada en `entregable/Docker/README_AUTOMATIZACION.md`.

## 13. Mantenimiento recomendado

Antes de cada nueva version se recomienda:

1. Validar el esquema y los catalogos del nuevo Excel.
2. Comparar la distribucion de variables contra el entrenamiento.
3. Reentrenar con una semilla y configuracion registradas.
4. Revisar ROC-AUC, F1, precision, recall, calibration y brechas por subgrupo.
5. Actualizar este documento y `metadata_modelo.json`.
6. Probar la API y construir la imagen Docker.
7. Mantener una version anterior del modelo para poder comparar o revertir.

# 🩻 Monitor de dosis en TC de cráneo pediátrico

Proyecto Final — Módulo 7

## Problema
En tomografía computada pediátrica la dosis de radiación debe ser la mínima necesaria: los niños son más
radiosensibles y tienen más años por delante para que aparezca un efecto tardío. Hoy la verificación de dosis la hace
un físico médico revisando estudios a mano, y los errores de configuración (un `mAs` mal tipeado, un protocolo
equivocado, un examen mal etiquetado) se detectan tarde o nunca.

**Para quién:** servicios de radiología pediátrica y sus físicos médicos / responsables de protección radiológica.

## Objetivo
Predecir la **dosis esperada (`CTDIvol_mGy`)** de un estudio a partir del paciente (altura, peso, edad, sexo) y del
protocolo (`kVp`, `mAs`, tipo de examen). Si la dosis registrada no es coherente con la esperada, o la técnica está
fuera de rango, el estudio se marca **REVISAR** para el físico médico.

**¿Por qué ML?** Una regla fija ("alertar si CTDIvol > X") no sirve: la dosis normal cambia según el examen y la
técnica (10 mGy es normal en un recién nacido con GG-Hel y un error grosero en un Dualscan). El modelo aprende la
relación protocolo + paciente → dosis y el residuo (observado − esperado) funciona como señal de anomalía.

### Métricas de éxito
| Métrica | Por qué | Meta | Resultado |
|---|---|---|---|
| MAE en test (mGy) | Mismas unidades que la dosis, interpretable por un físico médico | < 1 mGy | **0.067 mGy** (baseline 12.18) |
| Detección de errores simulados (mAs ×10 / dosis ×2) | Lo que importa en la práctica: no dejar pasar errores | ≥ 90 % | **100 % / 97.8 %** |
| Falsas alarmas en estudios correctos | Si alerta de más, nadie le hace caso | ≤ 5 % | **2.2 %** |

## Datos
Registros de TC de cráneo pediátrico (planilla de Kaggle en portugués): 722 filas crudas → **678 estudios de cráneo**
tras la limpieza (676 con dosis válida). Variables: altura, peso, edad, grupo del phantom (sexo), tipo de examen
(Dualscan / GG-Hel / Helical), kVp, mAs, DLP y CTDIvol.

Limpieza (detallada en el notebook, sección 2): filas vacías y de tórax, coma decimal, edades en meses/días,
alturas en metros, un peso con coma perdida, `mAs` 720 → 72, exámenes mal etiquetados y CTDIvol incoherentes con el DLP.

`datos_muestra.csv` contiene **100 filas** del dataset crudo (incluye los 4 estudios Helical). El dataset completo no
se incluye en la entrega; si el archivo `Dados_TC_cranio_kaggle.csv` no está, el notebook usa la muestra automáticamente.

## Modelo
Pipeline de scikit-learn guardado en `model.pkl`:

```
ColumnTransformer
 ├─ numéricas  (Altura_cm, Peso_kg, Idade_anos, kVp, mAs) → SimpleImputer(median) → StandardScaler
 └─ categóricas (Parametro_exame, Grupo_phantom)          → SimpleImputer(most_frequent) → OneHotEncoder
→ RandomForestRegressor(n_estimators=300)
```

Comparación con validación cruzada (5 folds, solo train):

| Modelo | MAE CV (mGy) | R² CV |
|---|---|---|
| Baseline (promedio) | 12.36 | −0.01 |
| Regresión Lineal | 0.224 | 0.995 |
| **Random Forest** (elegido, ajustado con GridSearchCV) | **0.136** | 0.994 |

`model.pkl` guarda un diccionario con el pipeline, los umbrales de alerta por examen (percentiles 95/99 de dosis,
límite del residuo calibrado con predicciones fuera de muestra y rango histórico de kVp/mAs) y las métricas de test.

## Resultados
* **Test (20 %, 136 estudios):** MAE 0.067 mGy · RMSE 0.61 mGy · R² 0.998.
* **Sistema de alertas en test:** detecta el 100 % de los `mAs` mal tipeados y el 97.8 % de las sobreexposiciones ×2,
  con 2.2 % de falsas alarmas.
* **Hallazgo clínico:** con solo altura, peso y edad el modelo no predice la dosis dentro de GG-Hel (R² −0.20, peor que
  la mediana). La dosis depende del tamaño del paciente **en escalones de técnica** (80 / 100 / 120 kVp), no de forma continua.

### ROI estimado (supuestos ilustrativos, editables en la app)
| Escenario | Estudios/mes | Beneficio anual | Costo anual | ROI |
|---|---|---|---|---|
| Conservador | 300 | USD 1.827 | USD 1.500 | 22 % |
| Base | 800 | USD 15.557 | USD 2.500 | 522 % |
| Optimista | 1.500 | USD 51.970 | USD 4.000 | 1.199 % |

Beneficio = horas de físico médico ahorradas + estudios repetidos evitados gracias a errores detectados.

## Estructura
```
├── README.md
├── notebook_final.ipynb   # pipeline completo, ejecutado
├── model.pkl              # pipeline + umbrales de alerta
├── datos_muestra.csv      # 100 filas del dataset crudo
├── app.py                 # demo Streamlit
└── requirements.txt
```

## Cómo ejecutar
```bash
pip install -r requirements.txt
jupyter notebook notebook_final.ipynb   # corre de arriba a abajo y regenera model.pkl
streamlit run app.py                    # demo: evaluar un estudio, un lote CSV y el ROI
```
> `model.pkl` fue generado con scikit-learn 1.8.0; con otra versión puede no cargar. En ese caso, correr el notebook
> para regenerarlo (con la muestra de 100 filas las métricas son menos estables que con el dataset completo).

## Próximos pasos
1. Validar con datos de otro tomógrafo/centro y con errores reales etiquetados por un físico médico.
2. Integrar con PACS/RIS leyendo el DICOM Radiation Dose Structured Report para evaluar cada estudio automáticamente.
3. Extender a otras regiones (tórax, abdomen) y predecir también el DLP.

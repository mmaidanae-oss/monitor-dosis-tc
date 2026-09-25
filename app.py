"""
App Streamlit — Monitor de dosis en TC de cráneo pediátrico.

Ejecutar:  streamlit run app.py
Requiere model.pkl (generado por notebook_final.ipynb) en la misma carpeta.
"""
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Monitor de dosis TC pediátrica", page_icon="🩻", layout="wide")


@st.cache_resource
def cargar_modelo(ruta="model.pkl"):
    return joblib.load(ruta)


def evaluar_estudio(estudio, art):
    """Misma lógica que en el notebook (sección 7)."""
    fila = pd.DataFrame([estudio])[art["features_num"] + art["features_cat"]]
    fila["Grupo_phantom"] = fila["Grupo_phantom"].astype(str)
    esperada = float(art["pipeline"].predict(fila)[0])
    obs = float(estudio[art["target"]])
    residuo = obs - esperada
    ref = art["umbrales"].get(estudio["Parametro_exame"])
    if ref is None:
        return {"veredicto": "REVISAR", "dosis_esperada": esperada, "residuo": residuo,
                "motivos": ["Tipo de examen no visto en el entrenamiento."]}
    motivos, veredicto = [], "OK"
    if obs > ref["p99_dosis"]:
        motivos.append(f"Dosis {obs:.2f} mGy > percentil 99 histórico del examen ({ref['p99_dosis']:.2f}).")
        veredicto = "REVISAR"
    elif obs > ref["p95_dosis"]:
        motivos.append(f"Dosis {obs:.2f} mGy > percentil 95 histórico del examen ({ref['p95_dosis']:.2f}).")
        veredicto = "ATENCION"
    for var in ["kVp", "mAs"]:
        lo, hi = ref[f"{var}_rango"]
        if not (lo <= float(estudio[var]) <= hi):
            motivos.append(f"{var}={estudio[var]:g} fuera del rango histórico de este examen ({lo:g}–{hi:g}): "
                           "posible error de tipeo o protocolo equivocado.")
            veredicto = "REVISAR"
    if abs(residuo) > ref["limite_residuo"]:
        motivos.append(f"Se esperaban ~{esperada:.2f} mGy para esta técnica y paciente; se registraron "
                       f"{obs:.2f} (residuo {residuo:+.2f}, límite ±{ref['limite_residuo']:.2f}).")
        veredicto = "REVISAR"
    if not motivos:
        motivos.append("Dosis coherente con el protocolo y el paciente.")
    return {"veredicto": veredicto, "dosis_esperada": esperada, "residuo": residuo, "motivos": motivos}


EJEMPLOS = {
    "GG-Hel típico (10 años)": dict(Altura_cm=121.0, Peso_kg=22.0, Idade_anos=10.0, Grupo_phantom="1",
                                    Parametro_exame="GG-Hel", kVp=100, mAs=187.0, CTDIvol_mGy=26.9),
    "Dualscan típico": dict(Altura_cm=110.0, Peso_kg=20.0, Idade_anos=6.0, Grupo_phantom="2",
                            Parametro_exame="Dualscan", kVp=80, mAs=72.0, CTDIvol_mGy=0.05),
    "Recién nacido, técnica reducida": dict(Altura_cm=50.0, Peso_kg=2.8, Idade_anos=0.01, Grupo_phantom="1",
                                            Parametro_exame="GG-Hel", kVp=80, mAs=80.0, CTDIvol_mGy=10.6),
    "ERROR: mAs tipeado 720": dict(Altura_cm=152.0, Peso_kg=39.0, Idade_anos=13.0, Grupo_phantom="1",
                                   Parametro_exame="Dualscan", kVp=80, mAs=720.0, CTDIvol_mGy=0.05),
    "ERROR: sobreexposición": dict(Altura_cm=100.0, Peso_kg=18.0, Idade_anos=5.0, Grupo_phantom="2",
                                   Parametro_exame="GG-Hel", kVp=100, mAs=187.0, CTDIvol_mGy=60.0),
}

# ---------------------------------------------------------------- cabecera
st.title("🩻 Monitor de dosis — TC de cráneo pediátrico")
st.caption("Predice la dosis esperada (CTDIvol) a partir del paciente y el protocolo, y marca los estudios "
           "cuya dosis registrada no es coherente para que los revise un físico médico.")

if not Path("model.pkl").exists():
    st.error("No se encontró model.pkl. Ejecutá primero notebook_final.ipynb.")
    st.stop()
art = cargar_modelo()
m = art.get("metricas_test", {})
if m:
    c1, c2, c3 = st.columns(3)
    c1.metric("MAE en test", f"{m['MAE']:.3f} mGy")
    c2.metric("RMSE en test", f"{m['RMSE']:.3f} mGy")
    c3.metric("R² en test", f"{m['R2']:.4f}")

tab1, tab2, tab3 = st.tabs(["Evaluar un estudio", "Evaluar un lote (CSV)", "ROI estimado"])

# ---------------------------------------------------------------- tab 1
with tab1:
    ejemplo = st.selectbox("Cargar un ejemplo", ["(ingresar a mano)"] + list(EJEMPLOS))
    base = EJEMPLOS.get(ejemplo, EJEMPLOS["GG-Hel típico (10 años)"])
    examenes = sorted(art["umbrales"])

    col_p, col_t = st.columns(2)
    with col_p:
        st.subheader("Paciente")
        altura = st.number_input("Altura (cm)", 30.0, 200.0, base["Altura_cm"], 1.0, key=f"alt{ejemplo}")
        peso = st.number_input("Peso (kg)", 1.0, 150.0, base["Peso_kg"], 0.1, key=f"pes{ejemplo}")
        edad = st.number_input("Edad (años; meses/12)", 0.0, 18.0, base["Idade_anos"], 0.01, key=f"eda{ejemplo}")
        sexo = st.radio("Grupo phantom", ["1", "2"], index=["1", "2"].index(base["Grupo_phantom"]),
                        format_func=lambda x: "1 · femenino" if x == "1" else "2 · masculino",
                        horizontal=True, key=f"sex{ejemplo}")
    with col_t:
        st.subheader("Protocolo y dosis registrada")
        examen = st.selectbox("Tipo de examen", examenes, index=examenes.index(base["Parametro_exame"]),
                              key=f"exa{ejemplo}")
        kvp = st.selectbox("kVp", [80, 100, 120], index=[80, 100, 120].index(base["kVp"]), key=f"kvp{ejemplo}")
        mas = st.number_input("mAs", 1.0, 2000.0, base["mAs"], 1.0, key=f"mas{ejemplo}")
        ctdi = st.number_input("CTDIvol registrado (mGy)", 0.0, 200.0, base["CTDIvol_mGy"], 0.01,
                               key=f"ctd{ejemplo}")

    if st.button("Evaluar estudio", type="primary"):
        estudio = dict(Altura_cm=altura, Peso_kg=peso, Idade_anos=edad, Grupo_phantom=sexo,
                       Parametro_exame=examen, kVp=kvp, mAs=mas, CTDIvol_mGy=ctdi)
        r = evaluar_estudio(estudio, art)
        a, b, c = st.columns(3)
        a.metric("Dosis esperada por el modelo", f"{r['dosis_esperada']:.2f} mGy")
        b.metric("Dosis registrada", f"{ctdi:.2f} mGy", delta=f"{r['residuo']:+.2f} mGy", delta_color="inverse")
        c.metric("Veredicto", r["veredicto"])
        mensaje = "\n".join(f"- {x}" for x in r["motivos"])
        {"OK": st.success, "ATENCION": st.warning, "REVISAR": st.error}[r["veredicto"]](mensaje)

# ---------------------------------------------------------------- tab 2
with tab2:
    st.write("Subí un CSV con las columnas: " + ", ".join(
        f"`{c}`" for c in art["features_num"] + art["features_cat"] + [art["target"]]))
    archivo = st.file_uploader("CSV de estudios", type="csv")
    if archivo is not None:
        lote = pd.read_csv(archivo).dropna(subset=art["features_num"] + art["features_cat"] + [art["target"]])
        res = [evaluar_estudio(f, art) for f in lote.to_dict("records")]
        lote = lote.assign(dosis_esperada=[round(x["dosis_esperada"], 2) for x in res],
                           residuo=[round(x["residuo"], 2) for x in res],
                           veredicto=[x["veredicto"] for x in res],
                           motivo=[" | ".join(x["motivos"]) for x in res])
        st.write(lote["veredicto"].value_counts())
        st.dataframe(lote.sort_values("veredicto", ascending=False), use_container_width=True)

# ---------------------------------------------------------------- tab 3
with tab3:
    st.write("Supuestos ilustrativos (USD). Ajustalos a tu servicio. Las tasas de detección y de falsas alarmas "
             "salen de la evaluación en test del notebook (sección 7).")
    fa = st.slider("Falsas alarmas del sistema", 0.0, 0.10, 0.022, 0.001, format="%.3f")
    det = st.slider("Tasa de detección de errores", 0.5, 1.0, 0.978, 0.001, format="%.3f")
    esc = pd.DataFrame({
        "estudios_mes": [300, 800, 1500], "min_revision_manual": [2.0, 3.0, 3.0],
        "fraccion_revisada_hoy": [0.30, 0.50, 1.00], "costo_hora_fisico": [15, 20, 30],
        "tasa_error_protocolo": [0.005, 0.01, 0.01], "costo_repeticion": [80, 120, 150],
        "costo_anual_sistema": [1500, 2500, 4000],
    }, index=["Conservador", "Base", "Optimista"])
    esc = st.data_editor(esc, use_container_width=True)
    horas_hoy = esc.estudios_mes * esc.fraccion_revisada_hoy * esc.min_revision_manual / 60
    horas_ml = esc.estudios_mes * (fa + esc.tasa_error_protocolo) * 5 / 60
    horas_ahorradas = (horas_hoy - horas_ml).clip(lower=0) * 12
    errores = esc.estudios_mes * esc.tasa_error_protocolo * det * 12
    beneficio = horas_ahorradas * esc.costo_hora_fisico + errores * esc.costo_repeticion
    roi = (beneficio - esc.costo_anual_sistema) / esc.costo_anual_sistema * 100
    tabla = pd.DataFrame({"horas físico ahorradas/año": horas_ahorradas.round(0),
                          "errores detectados/año": errores.round(0),
                          "beneficio anual USD": beneficio.round(0),
                          "costo anual USD": esc.costo_anual_sistema, "ROI %": roi.round(0)})
    st.dataframe(tabla, use_container_width=True)
    st.bar_chart(tabla["ROI %"])

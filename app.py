"""Streamlit interface for MotoTrack ABC/XYZ inventory classification."""

from __future__ import annotations

import hashlib
from io import BytesIO

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from classification import active_demand_window, build_abc_xyz_matrix, build_classification
from data_io import (
    DATA_DIR,
    DEMAND_META_PATH,
    DEMAND_PATH,
    PRICES_META_PATH,
    PRICES_PATH,
    WorkbookFormatError,
    initialise_sample_data,
    load_history,
    load_last_analysis,
    load_metadata,
    merge_demand_history,
    parse_demand_workbook,
    parse_prices_workbook,
    save_history,
    save_last_analysis,
    save_metadata,
)
from forecasting import run_backtesting


st.set_page_config(page_title="MotoTrack | ABC × XYZ", page_icon="📊", layout="wide")

st.markdown(
    """
    <style>
    :root { --ink: #172b2b; --muted: #5c7070; --accent: #087e75; --line: #d8e2df; }
    .block-container { padding-top: 1.5rem; padding-bottom: 3rem; }
    h1, h2, h3 { color: var(--ink); letter-spacing: 0; }
    [data-testid="stMetric"] { background: #f3f7f5; border-left: 3px solid var(--accent); padding: .7rem 1rem; }
    [data-testid="stSidebar"] { border-right: 1px solid var(--line); }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("MotoTrack · Clasificación ABC × XYZ")
st.caption("Contribución económica y confiabilidad del pronóstico por SKU")

prices_fixture_imported, demand_fixture_imported = initialise_sample_data()
st.session_state.setdefault("analysis_started", False)
if prices_fixture_imported or demand_fixture_imported:
    loaded_fixtures = []
    if prices_fixture_imported:
        loaded_fixtures.append("Precios_y_costos.xlsx")
    if demand_fixture_imported:
        loaded_fixtures.append("Demanda.xlsx")
    st.info("Datos de ejemplo inicializados desde: " + ", ".join(loaded_fixtures) + ".")

history = load_history()
price_metadata = load_metadata(PRICES_META_PATH)
demand_metadata = load_metadata(DEMAND_META_PATH)
last_exercise = load_last_analysis()
last_parameters = last_exercise.get("parameters", {}) if last_exercise else {}

with st.sidebar:
    st.header("Archivos")
    st.caption("Precios y costos se conservan en este equipo hasta una nueva carga.")
    prices_upload = st.file_uploader(
        "Actualizar precios y costos", type=["xlsx"], key="prices_uploader"
    )
    if prices_upload is not None:
        prices_bytes = prices_upload.getvalue()
        prices_hash = hashlib.sha256(prices_bytes).hexdigest()
        if st.session_state.get("processed_prices_hash") != prices_hash:
            try:
                parse_prices_workbook(BytesIO(prices_bytes))
                DATA_DIR.mkdir(parents=True, exist_ok=True)
                PRICES_PATH.write_bytes(prices_bytes)
                price_metadata = save_metadata(PRICES_META_PATH, prices_upload.name)
                st.session_state["processed_prices_hash"] = prices_hash
                st.session_state["analysis_started"] = False
                st.success("Precios y costos actualizados.")
            except WorkbookFormatError as error:
                st.session_state["processed_prices_hash"] = prices_hash
                st.error(str(error))

    if PRICES_PATH.exists():
        loaded_date = price_metadata.get("loaded_at", "fecha no disponible") if price_metadata else "fecha no disponible"
        source_name = price_metadata.get("original_name", PRICES_PATH.name) if price_metadata else PRICES_PATH.name
        st.caption(f"Activo: **{source_name}**\n\nCarga: {loaded_date}")
    else:
        st.warning("Carga un archivo con las hojas Costos y Precio venta.")

    st.divider()
    st.caption("Demanda se incorpora al histórico local por turno.")
    demand_upload = st.file_uploader("Incorporar demanda semanal", type=["xlsx"], key="demand_uploader")
    if demand_upload is not None:
        demand_bytes = demand_upload.getvalue()
        demand_hash = hashlib.sha256(demand_bytes).hexdigest()
        if st.session_state.get("seen_demand_hash") != demand_hash:
            try:
                incoming_demand = parse_demand_workbook(BytesIO(demand_bytes))
                st.session_state["pending_demand"] = {
                    "hash": demand_hash,
                    "name": demand_upload.name,
                    "data": incoming_demand,
                }
                st.session_state["seen_demand_hash"] = demand_hash
                st.session_state["analysis_started"] = False
            except WorkbookFormatError as error:
                st.session_state["seen_demand_hash"] = demand_hash
                st.session_state["pending_demand"] = None
                st.error(str(error))

    pending_demand = st.session_state.get("pending_demand")
    if pending_demand is not None:
        incoming_demand = pending_demand["data"]
        incoming_max = int(incoming_demand["turn"].max())
        existing_max = int(history["turn"].max()) if not history.empty else 0
        requires_confirmation = incoming_max <= existing_max
        st.caption(
            f"{pending_demand['name']}: turnos {int(incoming_demand['turn'].min())}–{incoming_max}. "
            f"Histórico actual: hasta {existing_max}."
        )
        confirmed = True
        if requires_confirmation:
            st.warning("El archivo no avanza el turno máximo guardado; podría ser antiguo.")
            confirmed = st.checkbox(
                "Confirmo incorporar este archivo y actualizar turnos existentes.",
                key="confirm_old_demand",
            )
        if not PRICES_PATH.exists():
            st.info("Carga primero el archivo de precios y costos para iniciar el análisis.")
        if st.button(
            "Iniciar aplicación",
            disabled=not confirmed or not PRICES_PATH.exists(),
            type="primary",
            width="stretch",
        ):
            history = merge_demand_history(history, incoming_demand)
            save_history(history)
            demand_metadata = save_metadata(DEMAND_META_PATH, pending_demand["name"])
            st.session_state["pending_demand"] = None
            st.session_state["analysis_started"] = True
            st.session_state.pop("confirm_old_demand", None)
            st.rerun()

    if demand_metadata:
        st.caption(
            f"Demanda activa: **{demand_metadata.get('original_name', 'histórico')}**\n\n"
            f"Carga: {demand_metadata.get('loaded_at', 'fecha no disponible')}"
        )
    elif not history.empty:
        st.caption(f"Histórico local: {DEMAND_PATH.name}")

    pending_demand = st.session_state.get("pending_demand")
    week_selection_history = history
    if pending_demand is not None:
        week_selection_history = merge_demand_history(history, pending_demand["data"])
    if not week_selection_history.empty:
        available_weeks = int(
            week_selection_history.groupby(["punto_venta", "producto"]).size().min()
        )
        default_weeks = int(last_parameters.get("analysis_weeks", available_weeks))
        selected_weeks = min(
            max(1, int(st.session_state.get("analysis_weeks", default_weeks))),
            available_weeks,
        )
        st.session_state["analysis_weeks"] = selected_weeks
        analysis_weeks = st.number_input(
            "Semanas a analizar",
            min_value=1,
            max_value=available_weeks,
            step=1,
            key="analysis_weeks",
            help=f"Puedes elegir entre 1 y {available_weeks} semanas disponibles por SKU.",
        )
        st.caption(f"Histórico disponible por SKU: hasta {available_weeks} semanas.")
        selected_window_size = int(analysis_weeks)
    else:
        analysis_weeks = 1
        selected_window_size = 1

    st.divider()
    st.header("Parámetros")
    abc_a_pct = st.number_input(
        "Corte A (%)", min_value=1.0, max_value=98.0,
        value=float(last_parameters.get("abc_a_pct", 80.0)), step=1.0, key="abc_a_pct"
    )
    abc_b_pct = st.number_input(
        "Corte B adicional (%)", min_value=1.0, max_value=98.0,
        value=float(last_parameters.get("abc_b_pct", 15.0)), step=1.0, key="abc_b_pct"
    )
    abc_c_pct = st.number_input(
        "Corte C (%)", min_value=1.0, max_value=98.0,
        value=float(last_parameters.get("abc_c_pct", 5.0)), step=1.0, key="abc_c_pct"
    )
    xyz_x_threshold = st.number_input(
        "Umbral X (Score%)", min_value=1.0, max_value=100.0,
        value=float(last_parameters.get("xyz_x_threshold", 80.0)), step=1.0, key="xyz_x_threshold"
    )
    xyz_y_threshold = st.number_input(
        "Umbral Y (Score%)", min_value=0.0, max_value=99.0,
        value=float(last_parameters.get("xyz_y_threshold", 60.0)), step=1.0, key="xyz_y_threshold"
    )
    max_test_weeks = max(1, selected_window_size - 2)
    test_weeks = st.number_input(
        "Semanas de prueba",
        min_value=1,
        max_value=max_test_weeks,
        value=min(max_test_weeks, max(1, int(last_parameters.get("test_weeks", 13)))),
        step=1,
        key="test_weeks",
    )
    training_weeks = selected_window_size - int(test_weeks)
    max_seasonal_period = max(
        2, min(selected_window_size // 2, max(2, training_weeks // 2))
    )
    seasonal_period = st.number_input(
        "Periodo estacional (semanas)",
        min_value=2,
        max_value=max_seasonal_period,
        value=min(max_seasonal_period, max(2, int(last_parameters.get("seasonal_period", 13)))),
        step=1,
        key="seasonal_period",
    )

if history.empty:
    st.info("Incorpora un archivo Demanda.xlsx para comenzar el análisis.")
    st.stop()

if not PRICES_PATH.exists():
    st.stop()

try:
    costs, prices = parse_prices_workbook(PRICES_PATH)
except WorkbookFormatError as error:
    st.error(str(error))
    st.stop()

pending_demand = st.session_state.get("pending_demand")
if not st.session_state["analysis_started"] and pending_demand is not None:
    st.info("Confirma el archivo de demanda en el panel lateral para iniciar la aplicación.")
    st.stop()

restoring_last_exercise = not st.session_state["analysis_started"] and last_exercise is not None
if not st.session_state["analysis_started"] and not restoring_last_exercise:
    st.info("Archivos listos. Inicia la aplicación para ejecutar el backtesting y la clasificación.")
    if st.button("Iniciar aplicación", type="primary", width="stretch", key="start_analysis"):
        st.session_state["analysis_started"] = True
        st.rerun()
    st.stop()

if restoring_last_exercise:
    saved_parameters = last_exercise["parameters"]
    saved_at = last_exercise.get("saved_at", "fecha no disponible")
    st.info(
        f"Mostrando el último ejercicio guardado ({saved_at}). "
        "Los cambios en parámetros se aplican al recalcular."
    )
    if st.button(
        "Recalcular con parámetros seleccionados",
        type="primary",
        width="stretch",
        key="recalculate_analysis",
    ):
        st.session_state["analysis_started"] = True
        st.rerun()

    classification = last_exercise["classification"]
    window_size = int(saved_parameters["analysis_weeks"])
    turn_start = int(saved_parameters["turn_start"])
    turn_end = int(saved_parameters["turn_end"])
    abc_a_pct = float(saved_parameters["abc_a_pct"])
    abc_b_pct = float(saved_parameters["abc_b_pct"])
    abc_c_pct = float(saved_parameters["abc_c_pct"])
    xyz_x_threshold = float(saved_parameters["xyz_x_threshold"])
    xyz_y_threshold = float(saved_parameters["xyz_y_threshold"])
    test_weeks = int(saved_parameters["test_weeks"])
    seasonal_period = int(saved_parameters["seasonal_period"])
    forecast_warnings = last_exercise.get("warnings", [])
    price_metadata = {
        "original_name": saved_parameters.get("price_file", "No disponible"),
        "loaded_at": saved_parameters.get("price_loaded_at", "No disponible"),
    }
    demand_metadata = {
        "original_name": saved_parameters.get("demand_file", "No disponible"),
        "loaded_at": saved_parameters.get("demand_loaded_at", "No disponible"),
    }
else:
    active_window = active_demand_window(history, weeks=int(analysis_weeks))
    turn_start = int(active_window["turn"].min())
    turn_end = int(active_window["turn"].max())
    sku_counts = active_window.groupby(["punto_venta", "producto"]).size()
    minimum_weeks = int(sku_counts.min())
    window_size = min(int(analysis_weeks), minimum_weeks)
    if minimum_weeks < int(analysis_weeks):
        st.warning(
            f"Hay menos de {int(analysis_weeks)} semanas para al menos un SKU; "
            f"se usan las {minimum_weeks} disponibles."
        )
    st.caption(f"Ventana de análisis: últimas {window_size} semanas · turnos {turn_start}–{turn_end}")

    if abs(abc_a_pct + abc_b_pct + abc_c_pct - 100) > 0.001:
        st.error("Los cortes A, B y C deben sumar 100%.")
        st.stop()
    if xyz_y_threshold >= xyz_x_threshold:
        st.error("El umbral Y debe ser menor que el umbral X.")
        st.stop()

    with st.spinner("Ejecutando backtesting y clasificación de los SKU..."):
        forecasts, forecast_warnings = run_backtesting(
            active_window,
            test_weeks=int(test_weeks),
            seasonal_period=int(seasonal_period),
        )
        try:
            classification = build_classification(
                active_window,
                costs,
                prices,
                forecasts,
                abc_a_pct=abc_a_pct,
                abc_b_pct=abc_b_pct,
                xyz_x_threshold=xyz_x_threshold,
                xyz_y_threshold=xyz_y_threshold,
                window_weeks=window_size,
            )
        except ValueError as error:
            st.error(str(error))
            st.stop()

    last_parameters = {
        "analysis_weeks": window_size,
        "abc_a_pct": float(abc_a_pct),
        "abc_b_pct": float(abc_b_pct),
        "abc_c_pct": float(abc_c_pct),
        "xyz_x_threshold": float(xyz_x_threshold),
        "xyz_y_threshold": float(xyz_y_threshold),
        "test_weeks": int(test_weeks),
        "seasonal_period": int(seasonal_period),
        "turn_start": turn_start,
        "turn_end": turn_end,
        "price_file": price_metadata.get("original_name", PRICES_PATH.name) if price_metadata else PRICES_PATH.name,
        "price_loaded_at": price_metadata.get("loaded_at", "No disponible") if price_metadata else "No disponible",
        "demand_file": demand_metadata.get("original_name", DEMAND_PATH.name) if demand_metadata else DEMAND_PATH.name,
        "demand_loaded_at": demand_metadata.get("loaded_at", "No disponible") if demand_metadata else "No disponible",
    }
    last_exercise = save_last_analysis(classification, last_parameters, forecast_warnings)

if forecast_warnings:
    with st.expander(f"Advertencias de pronóstico ({len(forecast_warnings)})"):
        for warning in forecast_warnings:
            st.warning(warning)

matrix = build_abc_xyz_matrix(classification)
utility_total = float(classification["Utilidad"].sum())
utility_column, summary_column = st.columns([2, 3])
utility_column.metric(
    "Utilidad total (COP)", f"$ {utility_total:,.0f}".replace(",", ".")
)
summary_metrics = summary_column.columns(3)
summary_metrics[0].metric("SKU clasificados", f"{len(classification)} / 8")
summary_metrics[1].metric("Ventana por SKU", f"{window_size} semanas")
summary_metrics[2].metric("Turnos", f"{turn_start}–{turn_end}")

results_tab, matrix_tab, trace_tab = st.tabs(["Resultados", "Matriz ABC × XYZ", "Trazabilidad"])
with results_tab:
    pareto_column, xyz_column = st.columns([1.4, 1])
    with pareto_column:
        pareto = make_subplots(specs=[[{"secondary_y": True}]])
        abc_colors = {"A": "#087e75", "B": "#df8a32", "C": "#9a5960"}
        pareto.add_trace(
            go.Bar(
                x=classification["sku"],
                y=classification["Utilidad"],
                marker_color=classification["ABC"].map(abc_colors),
                name="Utilidad",
                hovertemplate="%{x}<br>$%{y:,.0f}<extra></extra>",
            ),
            secondary_y=False,
        )
        pareto.add_trace(
            go.Scatter(
                x=classification["sku"],
                y=classification["% acumulado"],
                name="% acumulado",
                mode="lines+markers",
                line={"color": "#263d4d", "width": 2},
                marker={"size": 6},
                hovertemplate="%{x}<br>%{y:.1f}%<extra></extra>",
            ),
            secondary_y=True,
        )
        for cutoff in (abc_a_pct, abc_a_pct + abc_b_pct):
            pareto.add_hline(y=cutoff, line_dash="dash", line_color="#879694", secondary_y=True)
        pareto.update_layout(
            title="Pareto de utilidad",
            height=440,
            margin={"t": 60, "b": 135, "l": 20, "r": 20},
            legend={
                "orientation": "h",
                "x": 0.5,
                "xanchor": "center",
                "y": -0.42,
                "yanchor": "top",
                "bgcolor": "rgba(255,255,255,0)",
            },
        )
        pareto.update_xaxes(title_text="SKU", tickangle=-35)
        pareto.update_yaxes(title_text="Utilidad ($)", secondary_y=False)
        pareto.update_yaxes(title_text="Acumulado (%)", range=[0, 105], secondary_y=True)
        st.plotly_chart(pareto, width="stretch")

    with xyz_column:
        xyz_colors = {"X": "#087e75", "Y": "#df8a32", "Z": "#9a5960", "N/D": "#879694"}
        xyz_chart = go.Figure(
            go.Bar(
                x=classification["sku"],
                y=classification["Score%"],
                marker_color=classification["XYZ"].map(xyz_colors),
                customdata=classification[["RMSE", "XYZ"]],
                hovertemplate="%{x}<br>Score: %{y:.1f}%<br>RMSE: %{customdata[0]:.2f}<br>XYZ: %{customdata[1]}<extra></extra>",
            )
        )
        xyz_chart.add_hline(y=xyz_x_threshold, line_dash="dash", line_color="#087e75", annotation_text="X")
        xyz_chart.add_hline(y=xyz_y_threshold, line_dash="dash", line_color="#df8a32", annotation_text="Y")
        xyz_chart.update_layout(title="Score% de pronóstico", height=400, margin={"t": 55, "b": 80}, yaxis={"title": "Score%", "range": [0, 105]})
        xyz_chart.update_xaxes(title_text="SKU", tickangle=-35)
        st.plotly_chart(xyz_chart, width="stretch")

    st.subheader("Detalle por SKU")
    demand_column = f"Demanda {window_size} sem"
    display_columns = [
        "Punto de venta",
        "Producto",
        demand_column,
        "Precio venta",
        "Costo unitario",
        "Utilidad",
        "% utilidad",
        "% acumulado",
        "ABC",
        "Modelo elegido",
        "Score%",
        "MAE%",
        "RMSE",
        "XYZ",
    ]
    st.dataframe(
        classification[display_columns].style.format(
            {
                demand_column: "{:,.0f}",
                "Precio venta": "${:,.0f}",
                "Costo unitario": "${:,.0f}",
                "Utilidad": "${:,.0f}",
                "% utilidad": "{:.1f}%",
                "% acumulado": "{:.1f}%",
                "Score%": "{:.1f}%",
                "MAE%": "{:.1f}%",
                "RMSE": "{:.3f}",
            },
            na_rep="N/D",
        ),
        hide_index=True,
        width="stretch",
    )

    st.subheader("Conclusiones del ejercicio")
    abc_a_rows = classification[classification["ABC"] == "A"]
    abc_a_share = (
        abc_a_rows["Utilidad"].sum() / utility_total * 100 if utility_total else 0.0
    )
    top_sku = classification.iloc[0]
    xyz_counts = classification["XYZ"].value_counts()
    xyz_summary = " · ".join(
        f"{category}: {int(xyz_counts.get(category, 0))} SKU"
        for category in ("X", "Y", "Z")
    )
    x_count = int(xyz_counts.get("X", 0))
    y_count = int(xyz_counts.get("Y", 0))
    z_count = int(xyz_counts.get("Z", 0))
    most_common_xyz = xyz_counts.reindex(["X", "Y", "Z"]).fillna(0).idxmax()
    xyz_explanation = (
        f"Con los umbrales actuales (X ≥ {xyz_x_threshold:.0f}%, "
        f"Y entre {xyz_y_threshold:.0f}% y {xyz_x_threshold:.0f}%, "
        f"Z < {xyz_y_threshold:.0f}%), hay {x_count} SKU X, {y_count} SKU Y y {z_count} SKU Z. "
        f"X indica mejor ajuste en las semanas de prueba; Y, precisión intermedia; "
        f"Z, mayor incertidumbre. La clase más frecuente es {most_common_xyz}."
    )
    forecast_results = classification.dropna(subset=["RMSE"])
    if forecast_results.empty:
        forecast_summary = (
            "No hubo pronósticos válidos para comparar. Revisa las advertencias del backtesting "
            "y confirma que cada SKU tenga suficientes semanas."
        )
    else:
        best_forecast = forecast_results.loc[forecast_results["RMSE"].idxmin()]
        worst_forecast = forecast_results.loc[forecast_results["RMSE"].idxmax()]
        best_mae_pct = (
            f"{best_forecast['MAE%']:.1f}%"
            if pd.notna(best_forecast["MAE%"])
            else "N/D"
        )
        worst_mae_pct = (
            f"{worst_forecast['MAE%']:.1f}%"
            if pd.notna(worst_forecast["MAE%"])
            else "N/D"
        )
        forecast_summary = (
            f"El menor RMSE es {best_forecast['RMSE']:.2f} para {best_forecast['sku']} "
            f"(MAE% {best_mae_pct}); el mayor es {worst_forecast['RMSE']:.2f} para "
            f"{worst_forecast['sku']} (MAE% {worst_mae_pct}). RMSE expresa el error "
            "en unidades de demanda y penaliza más los errores grandes; MAE% expresa "
            "el error absoluto medio respecto al promedio real del periodo de prueba. "
            "En ambos casos, un valor menor significa un pronóstico más preciso."
        )

    st.markdown(
        "\n\n".join(
            [
                f"1. **Concentración de utilidad (ABC).** Los {len(abc_a_rows)} SKU "
                f"clasificados A generan el {abc_a_share:.1f}% de la utilidad total, "
                f"frente al corte A configurado de {abc_a_pct:.0f}%. Esta clase reúne "
                "los productos con mayor contribución económica acumulada; conviene "
                "darles prioridad en seguimiento y disponibilidad. La clasificación se "
                "basa en margen unitario por unidades vendidas durante las "
                f"{window_size} semanas seleccionadas.",
                f"2. **SKU de mayor aporte.** {top_sku['sku']} lidera con "
                f"$ {top_sku['Utilidad']:,.0f}".replace(",", ".")
                + f" ({top_sku['% utilidad']:.1f}% del total) y vendió "
                f"{top_sku[demand_column]:,.0f} unidades en el periodo seleccionado. "
                "Su aporte resulta de combinar las cantidades vendidas con el margen "
                "unitario; una variación de su disponibilidad puede tener un efecto "
                "importante en la utilidad agregada.",
                f"3. **Confiabilidad de pronóstico (XYZ).** {xyz_summary}. "
                + xyz_explanation
                + " Usa esta lectura junto con la importancia económica ABC: un SKU de "
                "alta contribución y baja confiabilidad merece revisión más frecuente "
                "de sus pronósticos; la clase XYZ por sí sola no mide rentabilidad.",
                f"4. **Desempeño del backtesting.** {forecast_summary} "
                f"La evaluación usa las últimas {int(test_weeks)} semanas como prueba "
                f"dentro de una ventana de {window_size} semanas. El mejor RMSE identifica "
                "el error más bajo entre modelos para un SKU, no garantiza el mismo "
                "resultado en semanas futuras.",
            ]
        )
    )

with matrix_tab:
    st.subheader("Matriz ABC × XYZ")
    st.dataframe(matrix.set_index("ABC / XYZ"), width="stretch")

with trace_tab:
    st.subheader("Parámetros y fuentes")
    st.write(
        {
            "Cortes ABC": f"A {abc_a_pct:.0f}% · B {abc_b_pct:.0f}% · C {abc_c_pct:.0f}%",
            "Umbrales XYZ": f"X ≥ {xyz_x_threshold:.0f}% · Y ≥ {xyz_y_threshold:.0f}% · Z < {xyz_y_threshold:.0f}%",
            "Periodo estacional": f"{int(seasonal_period)} semanas",
            "Semanas de prueba": int(test_weeks),
            "Semanas analizadas por SKU": window_size,
            "Ventana vigente": f"Turnos {turn_start}–{turn_end}",
            "Archivo de precios": price_metadata.get("original_name") if price_metadata else PRICES_PATH.name,
            "Carga de precios": price_metadata.get("loaded_at") if price_metadata else "No disponible",
            "Archivo de demanda": demand_metadata.get("original_name") if demand_metadata else DEMAND_PATH.name,
            "Carga de demanda": demand_metadata.get("loaded_at") if demand_metadata else "No disponible",
        }
    )


def build_excel_report() -> bytes:
    output = BytesIO()
    parameters = pd.DataFrame(
        [
            ("Corte A (%)", abc_a_pct),
            ("Corte B adicional (%)", abc_b_pct),
            ("Corte C (%)", abc_c_pct),
            ("Umbral X (Score%)", xyz_x_threshold),
            ("Umbral Y (Score%)", xyz_y_threshold),
            ("Periodo estacional (semanas)", int(seasonal_period)),
            ("Semanas de prueba", int(test_weeks)),
            ("Semanas analizadas por SKU", window_size),
            ("Ventana de turnos", f"{turn_start}–{turn_end}"),
            ("Archivo de precios", price_metadata.get("original_name") if price_metadata else PRICES_PATH.name),
            ("Carga de precios", price_metadata.get("loaded_at") if price_metadata else "No disponible"),
            ("Archivo de demanda", demand_metadata.get("original_name") if demand_metadata else DEMAND_PATH.name),
            ("Carga de demanda", demand_metadata.get("loaded_at") if demand_metadata else "No disponible"),
        ],
        columns=["Parámetro", "Valor"],
    )
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        classification[display_columns].to_excel(writer, sheet_name="Clasificacion", index=False)
        matrix.to_excel(writer, sheet_name="Matriz_ABC_XYZ", index=False)
        parameters.to_excel(writer, sheet_name="Parametros", index=False)
    return output.getvalue()


st.download_button(
    "Descargar reporte Excel",
    data=build_excel_report(),
    file_name="reporte_abc_xyz.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    type="primary",
)
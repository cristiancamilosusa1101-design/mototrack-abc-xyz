# MotoTrack ABC/XYZ

Aplicación local en Streamlit para clasificar los ocho SKU por contribución económica (ABC) y confiabilidad del pronóstico (XYZ).

## Requisitos

- Python 3.11 o superior
- Windows, macOS o Linux

## Instalación y ejecución

En Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

La primera ejecución inicializa los ejemplos de `samples/`. Streamlit muestra la dirección local en la terminal.

## Uso

- El archivo `Precios_y_costos.xlsx` activo se conserva en `data/precios_y_costos_actual.xlsx`; una carga válida lo reemplaza y registra nombre y fecha.
- Cada archivo semanal `Demanda.xlsx` se incorpora por upsert a `data/historico_demanda.csv`. Si su turno máximo no avanza el histórico, la app pide confirmación antes de aplicarlo.
- Después de cargar los dos archivos, pulsa **Iniciar aplicación** para incorporar la demanda pendiente y ejecutar el análisis. Si ya hay archivos activos sin una carga pendiente, el botón aparece en la pantalla principal.
- Al completar un ejercicio, la tabla, parámetros, fuentes y advertencias se guardan en `data/ultimo_ejercicio.json`. Al volver a abrir la app se muestra ese último resultado; **Recalcular con parámetros seleccionados** ejecuta uno nuevo y reemplaza el guardado anterior.
- Antes de iniciar, configura `Semanas a analizar` en el panel lateral; el máximo corresponde al histórico disponible para todos los SKU y el valor inicial usa todo ese histórico. ABC calcula la utilidad con las cantidades vendidas en el periodo elegido y XYZ ejecuta el backtesting sobre esa misma ventana. El detalle muestra el total de demanda con la duración seleccionada. El backtesting compara promedio móvil simple, suavización exponencial simple, regresión lineal, Holt-Winters y MSTL; elige el menor RMSE y calcula `Score% = max(0, 1 - MAE / promedio real) × 100`.
- Los controles permiten ajustar los cortes ABC/XYZ, el periodo estacional y las semanas de prueba. Las conclusiones contextualizan la concentración económica, el peso del SKU líder, la distribución XYZ y los errores de pronóstico; el periodo estacional no puede superar la mitad de la ventana.
- El botón de descarga genera un Excel con `Clasificacion`, `Matriz_ABC_XYZ` y `Parametros`.

Los archivos `data/` son estado local persistente y no deben compartirse como código fuente. Para reiniciar el estado de prueba, detén la app y elimina esa carpeta; la siguiente ejecución volverá a importar los fixtures.

## Pruebas

```powershell
python -m unittest discover -s tests
```
# Guía paso a paso para iniciar el proyecto

Esta guía te llevará desde cero hasta tener el sistema de trading funcionando.  
Incluye la instalación, configuración y ejecución del pipeline que genera señales de acciones listas para analizar o ejecutar manualmente.

---

## Instalación del entorno

1. Clonar el repositorio:
   ```bash
   git clone https://github.com/HankWht/HWTradingStrategy.git
   cd HWTradingStrategy
   ```

2. Crear y activar el entorno virtual (opcional, recomendado):
   ```bash
   python -m venv venv
   source venv/bin/activate        # Linux/macOS
   venv\Scripts\activate         # Windows
   ```

3. Instalar todas las dependencias necesarias (incluye Streamlit y Plotly):
   ```bash
   pip install -r requirements.txt
   ```

---

## Flujo completo de trabajo

### 1. Entrenar modelos y generar señales
Ejecuta el pipeline para descargar datos, construir features, entrenar los modelos y generar señales actuales:

```bash
python run_pipeline.py
```

Esto creará:
- Modelos entrenados en `/models`
- Señales actuales en `/reports/signals_latest.csv`

---

### 2. Ejecutar el backtest multi-asset
Simula trades históricos con los modelos entrenados para evaluar el rendimiento general:

```bash
python run_backtest.py
```

Genera:
- Curva de capital: `/reports/equity_curve.csv`
- Registro de trades: `/reports/trades_log.csv`

---

### 3. Lanzar el dashboard interactivo
Visualiza los resultados, rendimiento y señales desde un panel interactivo:

```bash
streamlit run dashboard.py
```

Esto abrirá automáticamente el navegador en:  
[http://localhost:8501](http://localhost:8501)

Podrás visualizar:

- Curva de capital (equity curve)
- Métricas clave del backtest (retorno, drawdown, win rate, profit factor)
- Historial de trades filtrable por ticker
- Señales actuales
- Rendimiento total por ticker
- Botón para recargar datos sin reiniciar Streamlit

---

## Dependencias principales

El archivo `requirements.txt` incluye todas las librerías necesarias:

```text
pandas==2.2.2
numpy==1.26.4
scikit-learn==1.5.2
xgboost==2.1.1
yfinance==0.2.40
matplotlib==3.9.2
pyyaml==6.0.2
joblib==1.4.2
streamlit==1.39.0
plotly==5.24.1
```

---

## Resultados esperados

Después de correr `run_backtest.py` o el notebook `03_backtest_and_analysis.ipynb`, deberías obtener resultados como:

```
{'final_capital': 10785.6,
 'total_return_pct': 7.85,
 'num_trades': 154,
 'win_rate_pct': 59.3,
 'avg_trade_pct': 0.17,
 'profit_factor': 1.72,
 'sharpe_like': 2.01,
 'max_drawdown_pct': -3.4}
```

---

## Paso 7 — Ver las señales generadas

Una vez completado el pipeline, abre el archivo:

```
reports/signals_latest.csv
```

Ejemplo de salida:

| Ticker | Probabilidad | Señal | Confianza | Precio actual | TP_target_% | SL_stop_% |
|--------|---------------|--------|-------------|----------------|--------------|-------------|
| NVDA | 0.74 | BUY | Alta | 113.52 | 0.7 | 0.25 |
| AAPL | 0.68 | WATCH | Media | 227.91 | 0.7 | 0.25 |
| TSLA | 0.53 | HOLD | Baja | 235.87 | 0.7 | 0.25 |

**Interpretación:**
- **BUY (Alta):** oportunidad fuerte → entrada recomendada  
- **WATCH (Media):** observar antes de entrar  
- **HOLD (Baja):** no operar  

---

## Paso 8 — Reentrenar o actualizar señales

Puedes ejecutar el pipeline:
- Diario → si quieres señales frescas intradía.  
- Semanal → para reentrenar modelos con nuevos datos.

Para agregar más acciones:
1. Añádelas al archivo `data/tickers.txt`
2. Ejecuta nuevamente:
   ```bash
   python run_pipeline.py
   ```

---

## Paso 9 — Archivos generados

| Carpeta | Contenido | Propósito |
|----------|------------|------------|
| `data/raw/` | Datos descargados de Yahoo Finance | Base histórica |
| `data/processed/` | Datasets con features + etiquetas | Entrenamiento |
| `models/` | Modelos entrenados (.pkl) | Reutilización |
| `reports/` | Señales y resultados del backtest | Análisis y decisiones |

---

## Paso 10 — Próximos pasos recomendados

Una vez el sistema de acciones esté estable, puedes:
1. Integrar ejecución **paper trading** con Alpaca API.  
2. Extender el sistema al módulo de **opciones (spreads verticales)**.  
3. Mejorar la interfaz con **dashboards avanzados en Streamlit o Dash**.  

---

## Resumen rápido de comandos

```bash
# Clonar o crear estructura
git clone https://github.com/HankWht/HWTradingStrategy.git
cd HWTradingStrategy

# Instalar librerías
pip install -r requirements.txt

# Correr el pipeline
python run_pipeline.py

# Ver señales
cat reports/signals_latest.csv
```

---

## Autor

**Hank Wht**  
HWTradingStrategy — Quantitative Trading Framework

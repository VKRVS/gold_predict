from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, validator
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import json
import logging
from typing import List

# Configurar logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Iniciar FastAPI
app = FastAPI(title="Modelo de predicción de precios del oro",
              description="API para predecir precios del oro basado en datos históricos",
              version="1.0.0")

# Cargar el modelo y los escaladores
try:
    model = tf.keras.models.load_model("modelo_lstm_oro.keras")
    scaler_X = joblib.load('scaler_features.joblib')
    scaler_y = joblib.load('scaler_target.joblib')
    
    # Cargar el archivo de metadata
    with open("metadata.json", "r") as f:
        metadata = json.load(f)
    
    logger.info("Modelo y escaladores cargados correctamente")
except Exception as e:
    logger.error(f"Error al cargar el modelo o escaladores: {str(e)}")
    raise

# Definir el esquema de entrada de la API
class RequestData(BaseModel):
    gold_prices: List[float]
    dxy_values: List[float]
    inflacion: float
    Bonos_10y: List[float]  # Nueva columna para Bonos_10y
    Boll_Upper: List[float]  # Nueva columna para Boll_Upper
    Boll_Lower: List[float]  # Nueva columna para Boll_Lower
    
# Función para calcular el ratio oro/dólar
def calculate_ratio_oro_dolar(gold_prices, dxy_values):
    return np.array(gold_prices) / np.array(dxy_values)

# Preprocesar los datos de entrada
def preprocess_data(gold_prices, dxy_values, bonos_10y_values, inflacion):
    try:
        # Crear el DataFrame base con los datos originales
        df_input = pd.DataFrame({
            "Oro": gold_prices,
            "DXY": dxy_values,
            "Bonos_10y": bonos_10y_values,
            "TIP": [inflacion] * len(gold_prices)
        })

        # Verificar nulos
        if df_input.isnull().any().any():
            logger.error("Hay valores nulos en los datos de entrada")
            raise ValueError("Los datos de entrada contienen valores nulos")

        # --- Feature Engineering ---
        df_input['Ratio_Oro_Dolar'] = df_input['Oro'] / df_input['DXY']
        df_input['Inflacion_Imp'] = df_input['TIP'].pct_change(periods=21, fill_method=None)
        df_input['Tasa_Real'] = df_input['Bonos_10y'] - df_input['Inflacion_Imp']
        df_input['SMA_200'] = df_input['Oro'].rolling(window=min(200, len(df_input))).mean()

        # Bandas de Bollinger y volatilidad
        window = min(20, len(df_input))
        rolling_mean = df_input['Oro'].rolling(window)
        rolling_std = df_input['Oro'].rolling(window).std()
        df_input['Boll_Upper'] = rolling_mean.mean() + 2 * rolling_std
        df_input['Boll_Lower'] = rolling_mean.mean() - 2 * rolling_std
        df_input['Volatilidad'] = df_input['Boll_Upper'] - df_input['Boll_Lower']

        # Llenamos nulos generados por los rolling
        df_input.fillna(0, inplace=True)

        # Verificamos columnas requeridas
        expected_columns = set(metadata['features_used'])
        current_columns = set(df_input.columns)

        if not expected_columns.issubset(current_columns):
            missing = expected_columns - current_columns
            logger.error(f"Faltan columnas en los datos de entrada: {missing}")
            raise ValueError(f"Faltan columnas en los datos de entrada: {missing}")

        # Reordenamos según el orden esperado
        df_input = df_input[metadata['features_used']]

        # Escalamos
        scaled_features = scaler_X.transform(df_input)
        return scaled_features

    except Exception as e:
        logger.error(f"Error en el preprocesamiento de datos: {str(e)}")
        raise


# Endpoint para la predicción
@app.post("/predict")
async def predict(request_data: RequestData):
    try:
        logger.info(f"Recibida solicitud con {len(request_data.gold_prices)} días de datos")
        logger.info(f"Datos: {request_data.model_dump()}")
        
        # Preprocesar los datos de entrada
        scaled_input = preprocess_data(
            request_data.gold_prices,
            request_data.dxy_values,
            request_data.Bonos_10y,
            request_data.inflacion
        )
        
        # Tomamos los últimos 90 días de datos
        sequence_length = 90
        last_sequence = scaled_input[-sequence_length:].reshape((1, sequence_length, len(metadata['features_used'])))
        
        # Realizar la predicción
        prediction_scaled = model.predict(last_sequence)

        # Convertir la predicción a valores reales
        prediction_real = scaler_y.inverse_transform(prediction_scaled)

        # Asegurar que las predicciones sean tipo float de Python
        predictions = [float(p) for p in prediction_real[0]]

        # Calcular el intervalo de confianza basado en la volatilidad histórica
        volatility = np.std(request_data.gold_prices[-20:])
        confidence_interval = [f"±{volatility * 100 / value:.2f}%" for value in predictions]

        # Recomendación de cobertura
        hedge_recommendation = "SHORT DXY futures" if predictions[-1] > request_data.gold_prices[-1] else "LONG DXY futures"

        logger.info("Predicción realizada con éxito")
        return {
            "predictions": predictions,
            "confidence_interval": confidence_interval,
            "hedge_recommendation": hedge_recommendation,
            "current_price": float(request_data.gold_prices[-1]),
            "last_price": float(request_data.gold_prices[-1]),
            "price_change": float(predictions[-1] - request_data.gold_prices[-1]),
            "percent_change": float((predictions[-1] - request_data.gold_prices[-1]) / request_data.gold_prices[-1] * 100)
        }

    except Exception as e:
        logger.error(f"Error al realizar la predicción: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error en el servidor: {str(e)}")

# Endpoint de salud para verificar que el servicio está funcionando
@app.get("/health")
async def health_check():
    return {"status": "OK", "model_loaded": model is not None}

# Endpoint para obtener información sobre el modelo
@app.get("/model-info")
async def model_info():
    return {
        "model_type": "LSTM",
        "features_used": metadata['features_used'],
        "sequence_length": 90,
        "output_length": len(metadata.get('output_features', ['price'])),
        "training_data_range": metadata.get('training_data_range', 'No disponible')
    }
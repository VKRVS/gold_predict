import requests
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Obtener los datos reales del oro, índice DXY, bonos y TIPs
def get_data():
    # Calcular fechas para obtener al menos 300 días de datos (200 para SMA + 90 para secuencia + margen)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=1000)  # Suficientes días para tener SMA 200 y secuencia
    
    # Obtener los datos históricos usando exactamente los mismos tickers
    df_oro = yf.download("GC=F", start=start_date, end=end_date)
    df_dxy = yf.download("DX-Y.NYB", start=start_date, end=end_date)
    df_bonos = yf.download("^TNX", start=start_date, end=end_date)
    df_inflacion = yf.download("TIP", start=start_date, end=end_date)
    
    # Mostrar la información de los datos obtenidos
    print(f"Días de datos - Oro: {len(df_oro)}, DXY: {len(df_dxy)}, Bonos: {len(df_bonos)}, TIPS: {len(df_inflacion)}")
    
    # Filtrar solo columnas de cierre (exactamente igual al código original)
    oro = df_oro['Close']
    dxy = df_dxy['Close']
    bonos = df_bonos['Close']
    tip = df_inflacion['Close']
    
    # Unir las series
    df = pd.concat([oro, dxy, bonos, tip], axis=1)
    df.columns = ["Oro", "DXY", "Bonos_10y", "TIP"]
    
    # Feature engineering (exactamente igual al código original)
    # Ratio oro/dólar
    df['Ratio_Oro_Dolar'] = df['Oro'] / df['DXY']
    
    # Inflación implícita (variación mensual de TIP)
    df['Inflacion_Imp'] = df['TIP'].pct_change(periods=21, fill_method=None)
    
    # Tasa real: Bonos - inflación implícita
    df['Tasa_Real'] = df['Bonos_10y'] - df['Inflacion_Imp']
    
    # SMA 200 días
    df['SMA_200'] = df['Oro'].rolling(window=200).mean()
    
    # Bandas de Bollinger (20 días, 2 desviaciones estándar)
    window = 20
    rolling_mean = df['Oro'].rolling(window)
    rolling_std = df['Oro'].rolling(window).std()

    # Usar `.mean()` y `.std()` para calcular las Bandas de Bollinger
    df['Boll_Upper'] = rolling_mean.mean() + 2 * rolling_std
    df['Boll_Lower'] = rolling_mean.mean() - 2 * rolling_std
    df['Volatilidad'] = df['Boll_Upper'] - df['Boll_Lower']
    
    # Eliminamos filas con valores nulos
    df.dropna(inplace=True)
    
    print(f"Datos después del feature engineering: {len(df)} días")
    
    # Extraemos los valores para la API
    gold_prices = df['Oro'].tolist()
    dxy_values = df['DXY'].tolist()
    
    # Para la inflación, usamos el valor más reciente de TIP
    inflacion = df['TIP'].iloc[-1]
    
    # Asegurar que tenemos suficientes datos
    if len(df) < 90:
        raise ValueError(f"No hay suficientes datos después del preprocessing. Se requieren al menos 90 días, pero solo hay {len(df)} días.")
    
    return gold_prices, dxy_values, inflacion, df

# Enviar los datos a la API y obtener la predicción
def get_prediction(gold_prices, dxy_values, inflacion, df):
    url = "http://127.0.0.1:8000/predict"

    # Adaptar para la API original
    data = {
        "gold_prices": gold_prices[-90:],  # Solo los últimos 90 días
        "dxy_values": dxy_values[-90:],    # Solo los últimos 90 días
        "inflacion": inflacion,
        "Bonos_10y": df['Bonos_10y'].iloc[-90:].tolist(),  # Enviar los últimos 90 días de bonos
        "Boll_Upper": df['Boll_Upper'].iloc[-90:].tolist(),  # Enviar los últimos 90 días de Boll_Upper
        "Boll_Lower": df['Boll_Lower'].iloc[-90:].tolist(),  # Enviar los últimos 90 días de Boll_Lower
    }
    print(f"Columnas de entrada a la API: {list(data.keys())}")

    print("Datos enviados a la API:")
    print(f"Longitud gold_prices: {len(data['gold_prices'])}")
    print(f"Longitud dxy_values: {len(data['dxy_values'])}")
    print(f"inflacion: {data['inflacion']}")
    print(f"Últimos 5 valores de gold_prices: {data['gold_prices'][-5:]}")
    print(f"Últimos 5 valores de dxy_values: {data['dxy_values'][-5:]}")
    
    try:
        response = requests.post(url, json=data)
        
        # Imprimir la respuesta completa
        print(f"Código de estado: {response.status_code}")
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error en la respuesta de la API: {response.text}")
            return {"error": "Error al obtener la predicción", "details": response.text}
    except requests.exceptions.RequestException as e:
        print(f"Error en la solicitud: {e}")
        return {"error": "Error al obtener la predicción"}


# Ejecutar la obtención de datos y predicción
try:
    gold_prices, dxy_values, inflacion, df = get_data()
    prediction = get_prediction(gold_prices, dxy_values, inflacion, df)
    
    # Mostrar la predicción
    if 'predictions' in prediction:
        print(f"Predicción de precio del oro: {prediction['predictions']}")
        print(f"Intervalo de confianza: {prediction['confidence_interval']}")
        print(f"Recomendación de cobertura: {prediction['hedge_recommendation']}")
    else:
        print("La respuesta de la API no contiene la clave 'predictions'. Aquí está la respuesta completa:")
        print(prediction)
except Exception as e:
    print(f"Error en la ejecución: {str(e)}")
    import traceback
    traceback.print_exc()
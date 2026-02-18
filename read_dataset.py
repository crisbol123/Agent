import pandas as pd
import json
import os

# Ruta del dataset
DATASET_PATH = r"D:\tesis\Agent\requirements_questions_v2.csv"

def load_dataset():
    """
    Carga el dataset requirements_questions_v2.csv
    """
    if not os.path.exists(DATASET_PATH):
        print(f"❌ Error: No se encontró el archivo {DATASET_PATH}")
        return None
    
    try:
        # Leer CSV
        df = pd.read_csv(DATASET_PATH)
        print(f"✅ Dataset cargado exitosamente")
        print(f"Filas: {len(df)}")
        print(f"Columnas: {list(df.columns)}")
        print()
        return df
    except Exception as e:
        print(f"❌ Error al leer el dataset: {str(e)}")
        return None

def show_sample(df, rows=5):
    """
    Muestra una muestra del dataset
    """
    if df is None:
        return
    
    print(f"📋 Primeras {rows} filas del dataset:")
    print(df.head(rows))
    print()

def show_info(df):
    """
    Muestra información del dataset
    """
    if df is None:
        return
    
    print("📊 Información del dataset:")
    print(df.info())
    print()

def get_row(df, index):
    """
    Obtiene una fila específica
    """
    if df is None or index >= len(df):
        return None
    
    return df.iloc[index]

if __name__ == "__main__":
    # Cargar dataset
    df = load_dataset()
    
    if df is not None:
        # Mostrar información
        show_info(df)
        show_sample(df, rows=3)
        
        # Ejemplo: acceder a la primera fila
        print("📌 Primera fila:")
        print(get_row(df, 0))

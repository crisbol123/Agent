import pandas as pd
from pathlib import Path

INPUT_FILE = "dataset_evaluacion_300_calificado.xlsx"
OUTPUT_FILE = "dataset_evaluacion_300_ordenado.xlsx"


def main() -> None:
    path = Path(INPUT_FILE)
    if not path.exists():
        raise FileNotFoundError(f"Archivo no encontrado: {INPUT_FILE}")

    # Detectar formato por extensión
    if path.suffix.lower() == '.xlsx' or 'xlsx' in path.name:
        df = pd.read_excel(path, engine='openpyxl')
        output_format = 'xlsx'
    else:
        # Auto-detectar delimitador para CSV
        with open(path, 'r', encoding='utf-8') as f:
            first_line = f.readline()
            sep = ';' if ';' in first_line else ','
        df = pd.read_csv(path, sep=sep, engine='python', on_bad_lines='skip')
        output_format = 'csv'

    if 'source' not in df.columns:
        raise ValueError("Falta columna 'source' en el archivo")

    # Separar coincidencias y desacuerdos
    df_coincidencias = df[df['source'] == 'coincidencia']
    df_desacuerdos = df[df['source'] == 'desacuerdo']

    # Concatenar: primero coincidencias, luego desacuerdos
    df_ordenado = pd.concat([df_coincidencias, df_desacuerdos], ignore_index=True)

    # Guardar en mismo formato que entrada
    if output_format == 'xlsx':
        df_ordenado.to_excel(OUTPUT_FILE, index=False, engine='openpyxl')
    else:
        df_ordenado.to_csv(OUTPUT_FILE, index=False, sep=sep)

    # Resumen
    print("=" * 70)
    print("DATASET ORDENADO")
    print("=" * 70)
    print(f"Archivo entrada: {INPUT_FILE}")
    print(f"Archivo salida: {OUTPUT_FILE}")
    print(f"Total muestras: {len(df_ordenado)}")
    print(f"\nPrimeras filas (coincidencias): {len(df_coincidencias)}")
    print(f"Últimas filas (desacuerdos): {len(df_desacuerdos)}")
    print("=" * 70)


if __name__ == "__main__":
    main()

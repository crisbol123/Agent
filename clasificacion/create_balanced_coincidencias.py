import pandas as pd
from pathlib import Path

INPUT_FILE = "comparacion_coincidencias.csv"
OUTPUT_FILE = "dataset_balanced_evaluation.csv"
SAMPLES_PER_CATEGORY = 70
RANDOM_STATE = 42

source = Path(INPUT_FILE)
if not source.exists():
    raise FileNotFoundError(f"No se encontro {INPUT_FILE}")

df = pd.read_csv(source)
required_cols = {"id", "question", "category"}
missing = required_cols - set(df.columns)
if missing:
    raise ValueError(f"Faltan columnas requeridas: {sorted(missing)}")

counts = df["category"].value_counts().sort_index()
min_count = int(counts.min())
if min_count < SAMPLES_PER_CATEGORY:
    raise ValueError(
        f"No hay suficientes muestras en al menos una categoria. Minimo disponible: {min_count}"
    )

# Muestreo balanceado por categoria (bucle explicito para compatibilidad)
samples = []
for category in sorted(df["category"].unique()):
    chunk = df[df["category"] == category].sample(
        n=SAMPLES_PER_CATEGORY,
        random_state=RANDOM_STATE,
    )
    samples.append(chunk)

df_balanced = (
    pd.concat(samples, ignore_index=True)
      .sample(frac=1, random_state=RANDOM_STATE)
      .reset_index(drop=True)
)

# Renombrar para compatibilidad con el evaluador actual
df_balanced = df_balanced.rename(columns={"category": "ground_truth_category"})

# Columna answer no existe en coincidencias; se agrega vacia para compatibilidad
df_balanced["answer"] = ""

# Orden de columnas esperado
df_balanced = df_balanced[["id", "question", "answer", "ground_truth_category"]]

df_balanced.to_csv(OUTPUT_FILE, index=False)

print("Dataset balanceado creado")
print(f"Archivo: {OUTPUT_FILE}")
print(f"Total: {len(df_balanced)}")
print("Distribucion por categoria:")
print(df_balanced["ground_truth_category"].value_counts().sort_index().to_string())

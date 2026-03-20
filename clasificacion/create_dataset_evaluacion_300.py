import pandas as pd
from pathlib import Path

SEED = 42
SAMPLES_PER_CATEGORY = 25
DISAGREE_FROM_OLLAMA = 12
DISAGREE_FROM_OPENAI = 13
CATEGORIES = ["ROUTING", "SECURITY", "QOS", "CONNECTIVITY", "MONITORING", "GENERAL"]

COINCIDENCIAS_FILE = "comparacion_coincidencias.csv"
DESACUERDOS_FILE = "comparacion_desacuerdos.csv"
OUTPUT_FILE = "dataset_evaluacion_300.csv"
DECISION_COLUMN = "decision_manual"


def load_csv(file_path: str) -> pd.DataFrame:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Archivo no encontrado: {file_path}")

    # Auto-detectar delimitador
    with open(path, 'r', encoding='utf-8') as f:
        first_line = f.readline()
        sep = ';' if ';' in first_line else ','

    df = pd.read_csv(path, sep=sep, engine='python', on_bad_lines='skip')

    # Validacion minima de columnas
    required_cols = {"id", "question"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas en {file_path}: {sorted(missing)}")

    # Soporte para ambos esquemas: 'category' o 'ollama/openai'
    if "category" not in df.columns:
        if {"ollama", "openai"}.issubset(df.columns):
            # En desacuerdos no hay 'category'; usamos la etiqueta de ollama para estratificar.
            df = df.copy()
            df["category"] = df["ollama"]
        else:
            raise ValueError(
                f"{file_path} no tiene columna 'category' ni columnas 'ollama/openai' para derivarla."
            )

    return df


def validate_categories(df: pd.DataFrame, file_label: str) -> None:
    counts = df["category"].value_counts()
    for cat in CATEGORIES:
        available = int(counts.get(cat, 0))
        if available < SAMPLES_PER_CATEGORY:
            raise ValueError(
                f"{file_label}: la categoria '{cat}' tiene {available} muestras; "
                f"se requieren al menos {SAMPLES_PER_CATEGORY}."
            )


def validate_disagreement_split(df: pd.DataFrame) -> None:
    if not {"ollama", "openai"}.issubset(df.columns):
        raise ValueError(
            "comparacion_desacuerdos.csv debe contener columnas 'ollama' y 'openai'."
        )

    for cat in CATEGORIES:
        n_ollama = int((df["ollama"] == cat).sum())
        n_openai = int((df["openai"] == cat).sum())
        total_available = n_ollama + n_openai

        if total_available < SAMPLES_PER_CATEGORY:
            raise ValueError(
                f"desacuerdo: categoria '{cat}' tiene {total_available} en total "
                f"(ollama={n_ollama}, openai={n_openai}); "
                f"se requieren al menos {SAMPLES_PER_CATEGORY}."
            )


def stratified_sample(df: pd.DataFrame, source_value: str) -> pd.DataFrame:
    validate_categories(df, source_value)

    sampled_parts = []
    for cat in CATEGORIES:
        part = df[df["category"] == cat].sample(
            n=SAMPLES_PER_CATEGORY,
            random_state=SEED,
            replace=False,
        )
        sampled_parts.append(part)

    sampled = pd.concat(sampled_parts, ignore_index=True).copy()
    sampled["source"] = source_value
    return sampled


def stratified_sample_desacuerdos(df: pd.DataFrame) -> pd.DataFrame:
    validate_disagreement_split(df)

    sampled_parts = []

    for cat in CATEGORIES:
        pool_ollama = df[df["ollama"] == cat]
        pool_openai = df[df["openai"] == cat]

        n_openai_take = min(DISAGREE_FROM_OPENAI, len(pool_openai), SAMPLES_PER_CATEGORY)
        n_ollama_take = SAMPLES_PER_CATEGORY - n_openai_take

        if len(pool_ollama) < n_ollama_take:
            # Si ollama no alcanza, tomar mas de openai (si existe disponibilidad)
            deficit = n_ollama_take - len(pool_ollama)
            n_ollama_take = len(pool_ollama)
            n_openai_take = min(SAMPLES_PER_CATEGORY, n_openai_take + deficit)

        sample_ollama = pool_ollama.sample(
            n=n_ollama_take,
            random_state=SEED,
            replace=False,
        ).copy()
        sample_ollama["category"] = cat
        sample_ollama["sampled_by"] = "ollama"

        sample_openai = pool_openai.sample(
            n=n_openai_take,
            random_state=SEED,
            replace=False,
        ).copy()
        sample_openai["category"] = cat
        sample_openai["sampled_by"] = "openai"

        sampled_parts.extend([sample_ollama, sample_openai])

    sampled = pd.concat(sampled_parts, ignore_index=True).copy()
    sampled["source"] = "desacuerdo"
    return sampled


def main() -> None:
    # 1) Cargar archivos
    df_coincidencias = load_csv(COINCIDENCIAS_FILE)
    df_desacuerdos = load_csv(DESACUERDOS_FILE)

    # 2) Muestreo estratificado: 25 por categoria en cada fuente
    sampled_coincidencias = stratified_sample(df_coincidencias, "coincidencia")
    sampled_desacuerdos = stratified_sample_desacuerdos(df_desacuerdos)

    # 3) Unir
    final_df = pd.concat([sampled_coincidencias, sampled_desacuerdos], ignore_index=True)

    # 4) Barajar dataset final con semilla fija
    final_df = final_df.sample(frac=1, random_state=SEED).reset_index(drop=True)

    # 5) Columna para decision del revisor
    final_df[DECISION_COLUMN] = ""

    # Orden recomendado para facilitar revision manual
    preferred_order = ["id", "question", "category", "source", DECISION_COLUMN]
    existing = [c for c in preferred_order if c in final_df.columns]
    remaining = [c for c in final_df.columns if c not in existing]
    final_df = final_df[existing + remaining]

    # 6) Guardar
    final_df.to_csv(OUTPUT_FILE, index=False)

    # 7) Resumen
    print("=" * 70)
    print("RESUMEN DATASET EVALUACION")
    print("=" * 70)
    print(f"Archivo generado: {OUTPUT_FILE}")
    print(f"Total muestras: {len(final_df)}")

    print("\nMuestras por categoria:")
    print(final_df["category"].value_counts().reindex(CATEGORIES, fill_value=0).to_string())

    print("\nMuestras por source:")
    print(final_df["source"].value_counts().to_string())

    print("=" * 70)


if __name__ == "__main__":
    main()

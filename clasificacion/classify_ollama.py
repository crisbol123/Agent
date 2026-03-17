"""
Clasificador semántico del dataset SLM_netconfig
Usa DeepSeek-R1 8B via Ollama (local, gratis)

Requisitos:
    1. Instalar Ollama: https://ollama.com
    2. Descargar modelo: ollama pull deepseek-r1:8b
    3. pip install ollama pandas tqdm

Uso:
    python classify_ollama.py                  # dataset completo
    python classify_ollama.py --sample 50      # prueba rápida
    python classify_ollama.py --model qwen2.5:14b
"""

import re
import json
import argparse
import pandas as pd
from pathlib import Path
from typing import Any
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

try:
    import ollama
except ImportError:
    raise SystemExit("Falta el SDK: pip install ollama")

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        print(f"Procesando {kwargs.get('total', '?')} registros...")
        return iterable


# ---------------------------------------------------------------------------
# Categorías
# ---------------------------------------------------------------------------

CAT_LIST = ["ROUTING", "SECURITY", "QOS", "CONNECTIVITY", "MONITORING", "GENERAL"]

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a Cisco network expert and dataset annotator.
Classify the given network configuration question into exactly one of these categories:

ROUTING, SECURITY, QOS, CONNECTIVITY, MONITORING, GENERAL

Rules:
- Choose the SINGLE most relevant category based on the question PRIMARY intent.
- If the question spans two domains, pick the one the question is MAINLY asking about.
- Use GENERAL when the question genuinely does not fit any specific technical domain.
- After your reasoning, respond with a JSON block. No extra text after the JSON.

Response format:
```json
{{"category": "CATEGORY_NAME"}}
```"""



# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------



def parse_json_from_response(text: str) -> dict | None:
    """Extrae el JSON de la respuesta aunque venga dentro de backticks."""
    # Buscar bloque ```json ... ```
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Buscar JSON suelto
    match = re.search(r"\{[^{}]*\"category\"[^{}]*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    return None


def build_run_checkpoint_path(base_path: str, model: str, input_path: str, sample: int | None) -> Path:
    base = Path(base_path)
    sample_tag = f"sample{sample}" if sample else "full"
    model_tag = re.sub(r"[^a-zA-Z0-9]+", "_", model).strip("_")
    input_tag = re.sub(r"[^a-zA-Z0-9]+", "_", Path(input_path).stem).strip("_")
    run_tag = f"{input_tag}_{model_tag}_{sample_tag}".lower()
    return base.with_name(f"{base.stem}_{run_tag}{base.suffix}")


# ---------------------------------------------------------------------------
# Clasificador
# ---------------------------------------------------------------------------

def single_call(model: str, question: str) -> str | None:
    """
    Una llamada al modelo.
    Retorna category.
    """
    try:
        response = ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": question},
            ],
            options={
                "temperature": 0.0,
                "num_predict": 128,
            },
        )
        raw = response["message"]["content"]
        print(f"\n--- MODEL ---\n{model}\n--- QUESTION ---\n{question}\n--- RAW RESPONSE ---\n{raw}\n--- END ---\n")
        parsed = parse_json_from_response(raw)

        if parsed and parsed.get("category") in CAT_LIST:
            return parsed["category"]

        for cat in CAT_LIST:
            if cat in raw.upper():
                return cat

        print(f"\n  [WARN] No se pudo parsear. Raw: {repr(raw[:200])}")
        return None

    except Exception as e:
        print(f"\n  Error en llamada: {e}")
        return None

    

def classify_question(model: str, question: str) -> dict:

    category = single_call(model, question)

    if not category:
        return {
            "category": "UNKNOWN",
        }

    return {
        "category": category,
    }


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------

def run(model: str, input_path: str, output_path: str, checkpoint_path: str,
    sample: int | None):

    df = pd.read_csv(input_path)
    if "id" not in df.columns:
        df.insert(0, "id", range(len(df)))

    if sample:
        df = df.sample(n=sample, random_state=42).reset_index(drop=True)
        print(f"Modo prueba: {sample} registros aleatorios.")

    total = len(df)
    print(f"\nDataset : {total} preguntas")
    print(f"Modelo  : {model} (Ollama local)")
    print("Clasificación: 1 llamada por pregunta\n")

    # Verificar que Ollama está corriendo
    try:
        ollama.list()
    except Exception:
        raise SystemExit(
            "No se puede conectar con Ollama.\n"
            "Asegúrate de que Ollama está corriendo: abre una terminal y ejecuta 'ollama serve'"
        )

    # Checkpoint
    checkpoint = build_run_checkpoint_path(checkpoint_path, model, input_path, sample)
    results: dict[int, dict[str, Any]] = {}
    print(f"Checkpoint: {checkpoint.name}")

    if checkpoint.exists():
        with open(checkpoint) as f:
            results = {int(k): v for k, v in json.load(f).items()}
        print(f"Checkpoint: {len(results)} registros ya clasificados.")

    all_ids = [int(x) for x in df["id"].tolist()]
    pending = [row_id for row_id in all_ids if row_id not in results]
    print(f"Pendientes: {len(pending)}\n")

    save_every = 50

    for count, row_id in enumerate(tqdm(pending, total=len(pending), desc="Clasificando")):
        row = df[df["id"] == row_id].iloc[0]
        question = str(row["question"])
        results[row_id] = classify_question(model, question)

        if (count + 1) % save_every == 0:
            with open(checkpoint, "w") as f:
                json.dump(results, f)
            unknown = sum(1 for r in results.values() if r.get("category") == "UNKNOWN")
            print(f"  [{len(results)}/{total}] guardado | UNKNOWN hasta ahora: {unknown}")

    # Checkpoint final
    with open(checkpoint, "w") as f:
        json.dump(results, f)

    # Construir DataFrame
    df["category"] = [results[int(row_id)]["category"] for row_id in df["id"]]

    df.to_csv(output_path, index=False)
    print(f"\nDataset clasificado: {len(df)} registros → {output_path}")

    unknown_count = int((df["category"] == "UNKNOWN").sum())
    print(f"UNKNOWN          : {unknown_count} registros")

    print("\n=== DISTRIBUCIÓN FINAL ===")
    stats = df["category"].value_counts()
    pct   = df["category"].value_counts(normalize=True) * 100
    print(pd.DataFrame({"N": stats, "%": pct.round(1)}))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Clasificador SLM_netconfig con Qwen via Ollama"
    )
    parser.add_argument("--model",      type=str, default="qwen2.5:14b")
    parser.add_argument("--input",      type=str, default="requirements_questions_v2.csv")
    parser.add_argument("--output",     type=str, default="requirements_classified_v3.csv")
    parser.add_argument("--checkpoint", type=str, default="ollama_checkpoint_v3.json")
    parser.add_argument("--sample",     type=int, default=None)
    args = parser.parse_args()

    run(
        model=args.model,
        input_path=args.input,
        output_path=args.output,
        checkpoint_path=args.checkpoint,
        sample=args.sample,
    )

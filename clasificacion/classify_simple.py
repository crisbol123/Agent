"""
Clasificador simple - 1 voto, 1 CSV de salida
El CSV de salida es igual al de entrada pero con la columna 'category' agregada.

Uso:
    python classify_simple.py
    python classify_simple.py --sample 20
    python classify_simple.py --model llama3.1:8b
"""

import re
import json
import argparse
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import ollama

CAT_LIST = ["ROUTING", "SECURITY", "QOS", "CONNECTIVITY", "MONITORING", "GENERAL"]

SYSTEM_PROMPT = """You are a Cisco network expert and dataset annotator.
Classify the given network configuration question into exactly one of these categories:

ROUTING, SECURITY, QOS, CONNECTIVITY, MONITORING, GENERAL

Rules:
- Choose the SINGLE most relevant category based on the question PRIMARY intent.
- If the question spans two domains, pick the one the question is MAINLY asking about.
- Use GENERAL only when the question genuinely does not fit any specific technical domain.
- After your reasoning, respond with a JSON block. No extra text after the JSON.

Response format:
```json
{"category": "CATEGORY_NAME", "reason": "one short sentence"}
```"""


def parse_json_from_response(text: str) -> dict | None:
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    match = re.search(r"\{[^{}]*\"category\"[^{}]*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    return None


def classify(model: str, question: str) -> str:
    try:
        response = ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": question},
            ],
            options={"temperature": 0.3, "num_predict": 2048},
        )
        raw = response["message"]["content"]
        print(f"\n--- QUESTION ---\n{question}\n--- RESPONSE ---\n{raw}\n--- END ---\n")

        parsed = parse_json_from_response(raw)
        if parsed and parsed.get("category") in CAT_LIST:
            return parsed["category"]

        for cat in CAT_LIST:
            if cat in raw.upper():
                return cat

        print(f"  [WARN] No se pudo parsear. Raw: {repr(raw[:200])}")
        return "UNKNOWN"

    except Exception as e:
        print(f"  [ERROR] {e}")
        return "UNKNOWN"


def run(model: str, input_path: str, output_path: str, checkpoint_path: str, sample: int | None):
    df = pd.read_csv(input_path)
    if sample:
        df = df.sample(n=sample, random_state=42).reset_index(drop=True)
        print(f"Modo prueba: {sample} registros aleatorios.")

    total = len(df)
    print(f"\nDataset : {total} preguntas")
    print(f"Modelo  : {model}\n")

    try:
        ollama.list()
    except Exception:
        raise SystemExit("No se puede conectar con Ollama. Ejecuta 'ollama serve'")

    checkpoint = Path(checkpoint_path)
    results: dict[int, str] = {}
    if checkpoint.exists():
        with open(checkpoint) as f:
            results = {int(k): v for k, v in json.load(f).items()}
        print(f"Checkpoint: {len(results)} registros ya clasificados.")

    pending = [i for i in range(total) if i not in results]
    print(f"Pendientes: {len(pending)}\n")

    for count, idx in enumerate(tqdm(pending, total=len(pending), desc="Clasificando")):
        question = str(df.loc[idx, "question"])
        results[idx] = classify(model, question)

        if (count + 1) % 50 == 0:
            with open(checkpoint, "w") as f:
                json.dump(results, f)

    with open(checkpoint, "w") as f:
        json.dump(results, f)

    df["category"] = [results[i] for i in range(total)]
    df.to_csv(output_path, index=False)
    print(f"\nDataset guardado: {output_path}")

    print("\n=== DISTRIBUCIÓN FINAL ===")
    print(df["category"].value_counts().to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",      type=str, default="llama3.1:8b")
    parser.add_argument("--input",      type=str, default="requirements_questions_v2.csv")
    parser.add_argument("--output",     type=str, default="requirements_classified_simple.csv")
    parser.add_argument("--checkpoint", type=str, default="checkpoint_simple.json")
    parser.add_argument("--sample",     type=int, default=None)
    args = parser.parse_args()

    run(
        model=args.model,
        input_path=args.input,
        output_path=args.output,
        checkpoint_path=args.checkpoint,
        sample=args.sample,
    )

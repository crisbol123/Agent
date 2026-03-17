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
    python classify_ollama.py --model deepseek-r1:8b  # modelo por defecto
"""

import re
import json
import time
import argparse
import pandas as pd
from pathlib import Path
from collections import Counter

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
{{"category": "CATEGORY_NAME", "reason": "one short sentence"}}
```"""



CONFIDENCE_THRESHOLD = 0.70


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


def compute_confidence( category: str, all_votes: list[str]) -> float:
    if not all_votes:
        return 0.0
    counter = Counter(all_votes)
    winner_count = counter[category]
    return round(winner_count / len(all_votes), 2)

# ---------------------------------------------------------------------------
# Clasificador
# ---------------------------------------------------------------------------

def single_call(model: str, question: str) -> tuple[str | None, str]:
    """
    Una llamada al modelo.
    Retorna (category, reason).
    """
    try:
        response = ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": question},
            ],
            options={
                "temperature": 0.7,
                "num_predict": 2048,
            },
        )
        raw = response["message"]["content"]
        print(f"\n--- QUESTION ---\n{question}\n--- RAW RESPONSE ---\n{raw}\n--- END ---\n")
        parsed = parse_json_from_response(raw)

        if parsed and parsed.get("category") in CAT_LIST:
            return parsed["category"], parsed.get("reason", "")

        for cat in CAT_LIST:
            if cat in raw.upper():
                return cat, "Extracted from response text"

        print(f"\n  [WARN] No se pudo parsear. Raw: {repr(raw[:200])}")
        return None, ""

    except Exception as e:
        print(f"\n  Error en llamada: {e}")
        return None, ""

    


def classify_question(model: str, question: str, n_votes: int = 2) -> dict:
  
    votes = []
    reasons = []
    

    for _ in range(n_votes):
        cat, reason = single_call(model, question)
        if cat:
            votes.append(cat)
            reasons.append(reason)
            

    if not votes:
        return {
            "category": "UNKNOWN",
            "confidence": 0.0,
            "reason": "All calls failed",
            "votes": [],
            "low_confidence": True,
        }

    # Categoría ganadora
    counter = Counter(votes)
    winner, _ = counter.most_common(1)[0]

 
    winning_reason = next(
        (r for v, r in zip(votes, reasons) if v == winner), reasons[0]
    )

    confidence = compute_confidence(winner, votes)
    low_conf = confidence < CONFIDENCE_THRESHOLD

    return {
        "category": winner,
        "confidence": confidence,
        "reason": winning_reason,
        "votes": votes,
        "low_confidence": low_conf,
    }


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------

def run(model: str, input_path: str, output_path: str,
        low_conf_path: str, checkpoint_path: str,
        sample: int | None, n_votes: int):

    df = pd.read_csv(input_path)
    if sample:
        df = df.sample(n=sample, random_state=42).reset_index(drop=True)
        print(f"Modo prueba: {sample} registros aleatorios.")

    total = len(df)
    print(f"\nDataset : {total} preguntas")
    print(f"Modelo  : {model} (Ollama local)")
    print(f"Votos   : {n_votes} por pregunta")
    print(f"Umbral  : confianza < {CONFIDENCE_THRESHOLD} → revisión manual\n")

    # Verificar que Ollama está corriendo
    try:
        ollama.list()
    except Exception:
        raise SystemExit(
            "No se puede conectar con Ollama.\n"
            "Asegúrate de que Ollama está corriendo: abre una terminal y ejecuta 'ollama serve'"
        )

    # Checkpoint
    checkpoint = Path(checkpoint_path)
    results: dict[int, dict] = {}
    if checkpoint.exists():
        with open(checkpoint) as f:
            results = {int(k): v for k, v in json.load(f).items()}
        print(f"Checkpoint: {len(results)} registros ya clasificados.")

    pending = [i for i in range(total) if i not in results]
    print(f"Pendientes: {len(pending)}\n")

    save_every = 50

    for count, idx in enumerate(tqdm(pending, total=len(pending), desc="Clasificando")):
        question = str(df.loc[idx, "question"])
        results[idx] = classify_question(model, question, n_votes)

        if (count + 1) % save_every == 0:
            with open(checkpoint, "w") as f:
                json.dump(results, f)
            low = sum(1 for r in results.values() if r.get("low_confidence"))
            print(f"  [{len(results)}/{total}] guardado | baja confianza hasta ahora: {low}")

    # Checkpoint final
    with open(checkpoint, "w") as f:
        json.dump(results, f)

    # Construir DataFrame
    df["category"]       = [results[i]["category"]                  for i in range(total)]
    df["confidence"]     = [results[i]["confidence"]                for i in range(total)]
    df["reason"]         = [results[i]["reason"]                    for i in range(total)]
    df["votes"]          = ["|".join(results[i].get("votes", []))   for i in range(total)]
    df["low_confidence"] = [results[i].get("low_confidence", False) for i in range(total)]

    # Separar confiables de dudosos
    clean_df = df[df["low_confidence"] == False].copy()
    low_df   = df[df["low_confidence"] == True].copy()

    # CSV solo clasificaciones confiables
    clean_df.to_csv(output_path, index=False)
    print(f"\nDataset clasificado: {len(clean_df)} registros → {output_path}")

    # CSV baja confianza / UNKNOWN
    low_df.to_csv(low_conf_path, index=False)
    print(f"Revisión manual    : {len(low_df)} registros → {low_conf_path}")

    # Resumen (solo confiables)
    print("\n=== DISTRIBUCIÓN FINAL (confiables) ===")
    stats = clean_df["category"].value_counts()
    pct   = clean_df["category"].value_counts(normalize=True) * 100
    print(pd.DataFrame({"N": stats, "%": pct.round(1)}))

    print(f"\nConfianza promedio : {clean_df['confidence'].mean():.3f}")
    print(f"Alta confianza (≥{CONFIDENCE_THRESHOLD}) : {len(clean_df)} ({len(clean_df)/total*100:.1f}%)")
    print(f"Revisión manual    : {len(low_df)} ({len(low_df)/total*100:.1f}%) → revisa manualmente")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Clasificador SLM_netconfig con DeepSeek-R1 via Ollama"
    )
    parser.add_argument("--model",      type=str, default="llama3.1:8b")
    parser.add_argument("--input",      type=str, default="requirements_questions_v2.csv")
    parser.add_argument("--output",     type=str, default="requirements_classified_v2.csv")
    parser.add_argument("--low-conf",   type=str, default="requirements_low_confidence_v2.csv")
    parser.add_argument("--checkpoint", type=str, default="ollama_checkpoint_v2.json")
    parser.add_argument("--sample",     type=int, default=None)
    parser.add_argument("--votes",      type=int, default=2,
                        help="Número de votos por pregunta (default: 2)")
    args = parser.parse_args()

    run(
        model=args.model,
        input_path=args.input,
        output_path=args.output,
        low_conf_path=args.low_conf,
        checkpoint_path=args.checkpoint,
        sample=args.sample,
        n_votes=args.votes,
    )

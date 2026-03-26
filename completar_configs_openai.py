import argparse
import json
import re
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

try:
    import ollama
except ImportError:
    raise SystemExit("Falta el SDK: pip install ollama")


def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def load_json(path: Path) -> dict:
    return json.loads(load_text(path))


def build_snapshot_context(snapshot_dir: Path) -> str:
    topo_path = snapshot_dir / "batfish" / "layer1_topology.json"
    h1_path = snapshot_dir / "hosts" / "h1.json"
    h2_path = snapshot_dir / "hosts" / "h2.json"

    r1_cfg = load_text(snapshot_dir / "configs" / "router1.cfg")
    r2_cfg = load_text(snapshot_dir / "configs" / "router2.cfg")
    r3_cfg = load_text(snapshot_dir / "configs" / "router3.cfg")
    r4_cfg = load_text(snapshot_dir / "configs" / "router4.cfg")

    topo = load_json(topo_path)
    h1 = load_json(h1_path)
    h2 = load_json(h2_path)

    return (
        "TOPOLOGY_LAYER1_JSON:\n"
        + json.dumps(topo, ensure_ascii=True, indent=2)
        + "\n\nHOST_H1_JSON:\n"
        + json.dumps(h1, ensure_ascii=True, indent=2)
        + "\n\nHOST_H2_JSON:\n"
        + json.dumps(h2, ensure_ascii=True, indent=2)
        + "\n\nROUTER1_RUNNING_CONFIG:\n"
        + r1_cfg
        + "\n\nROUTER2_RUNNING_CONFIG:\n"
        + r2_cfg
        + "\n\nROUTER3_RUNNING_CONFIG:\n"
        + r3_cfg
        + "\n\nROUTER4_RUNNING_CONFIG:\n"
        + r4_cfg
    )


def parse_json_block(text: str) -> dict | None:
    text = (text or "").strip()
    if not text:
        return None

    fenced = re.search(r"```(?:json)?\s*(\{[\s\S]*\})\s*```", text)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except json.JSONDecodeError:
            pass

    bare = re.search(r"\{.*\}", text, re.DOTALL)
    if bare:
        try:
            return json.loads(bare.group(0))
        except json.JSONDecodeError:
            return None

    return None


def normalize_answer(answer: str) -> str:
    txt = (answer or "").strip()
    if not txt:
        return "NO_CODE"

    if txt.upper() == "NO_CODE":
        return "NO_CODE"

    txt = re.sub(r"```[a-zA-Z]*\n?", "", txt)
    txt = txt.replace("```", "").strip()
    return txt if txt else "NO_CODE"


def complete_answer(
    model: str,
    num_predict: int,
    context_block: str,
    requirement: str,
    partial_answer: str,
) -> tuple[str, str, str]:
    system_prompt = """You are a Cisco IOS configuration completion assistant.
You must complete potentially truncated Cisco configurations using only the provided context.

Hard rules:
- Use only information available in the requirement and topology/context.
- Do not invent interfaces, links, IPs, device names, or protocols not supported by context.
- If the partial answer is already complete and valid for the requirement, keep it as-is.
- If information is insufficient to safely complete, return NO_CODE.
- Return ONLY a JSON object, no extra text.

Output JSON schema:
{
  "action": "KEEP_AS_IS" | "COMPLETED" | "NO_CODE",
  "completed_answer": "..."
}
"""

    user_prompt = (
        "REQUIREMENT:\n"
        + requirement
        + "\n\nEXISTING_PARTIAL_ANSWER:\n"
        + (partial_answer or "")
        + "\n\nNETWORK_CONTEXT:\n"
        + context_block
    )

    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        options={
            "temperature": 0.0,
            "num_predict": num_predict,
        },
    )

    raw = response["message"]["content"].strip()
    parsed = parse_json_block(raw)
    if not parsed:
        return "ERROR", "ERROR", raw

    action = str(parsed.get("action", "")).strip().upper()
    completed = normalize_answer(str(parsed.get("completed_answer", "")))

    if action == "KEEP_AS_IS":
        return action, normalize_answer(partial_answer), raw

    if action == "COMPLETED":
        return action, completed, raw

    if action == "NO_CODE":
        return action, "NO_CODE", raw

    return "ERROR", "ERROR", raw


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Completa respuestas truncadas de configuracion Cisco con Ollama y contexto de snapshot."
    )
    parser.add_argument("--input", default="generated_answers_cleaned.csv")
    parser.add_argument("--output", default="generated_answers_completed.csv")
    parser.add_argument("--snapshot", default="snapshot")
    parser.add_argument("--model", default="qwen2.5:14b")
    parser.add_argument("--num-predict", type=int, default=900)
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--sleep", type=float, default=0.0, help="Sleep in seconds between requests")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    snapshot_dir = Path(args.snapshot)

    if not input_path.exists():
        raise SystemExit(f"No existe input CSV: {input_path}")

    if not snapshot_dir.exists():
        raise SystemExit(f"No existe carpeta snapshot: {snapshot_dir}")

    try:
        ollama.list()
    except Exception:
        raise SystemExit(
            "No se puede conectar con Ollama. Ejecuta 'ollama serve' y verifica que el modelo exista."
        )

    df = pd.read_csv(input_path)
    required_cols = {"requirement", "model_answer"}
    missing = required_cols - set(df.columns)
    if missing:
        raise SystemExit(f"Faltan columnas en CSV: {missing}")

    if args.sample:
        df = df.sample(n=args.sample, random_state=42).reset_index(drop=True)

    context_block = build_snapshot_context(snapshot_dir)

    actions = []
    completed_answers = []
    raw_outputs = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Completando respuestas"):
        req = str(row.get("requirement", ""))
        partial = str(row.get("model_answer", ""))

        try:
            action, completed, raw = complete_answer(
                model=args.model,
                num_predict=args.num_predict,
                context_block=context_block,
                requirement=req,
                partial_answer=partial,
            )
        except Exception as exc:
            action, completed, raw = "ERROR", "ERROR", str(exc)

        actions.append(action)
        completed_answers.append(completed)
        raw_outputs.append(raw)

        if args.sleep > 0:
            time.sleep(args.sleep)

    out = df.copy()
    out["completion_action"] = actions
    out["model_answer_completed"] = completed_answers
    out["model_raw_output"] = raw_outputs

    out.to_csv(output_path, index=False)

    print("Proceso completado")
    print(f"Input : {input_path}")
    print(f"Output: {output_path}")
    print("\nAcciones:")
    print(out["completion_action"].value_counts(dropna=False))


if __name__ == "__main__":
    main()

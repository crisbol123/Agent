import os
import gc
import json
import time
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

warnings.filterwarnings("ignore")

# ── Entorno HuggingFace (igual que clasificacion) ─────────────────────────────
ENV_FILE = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=ENV_FILE)

HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError(f"No se encontro HF_TOKEN en {ENV_FILE}")

os.environ["HF_TOKEN"]              = HF_TOKEN
os.environ["HF_HOME"]               = r"D:\huggingface"
os.environ["HUGGINGFACE_HUB_CACHE"] = r"D:\huggingface\hub"
os.environ["TRANSFORMERS_CACHE"]    = r"D:\huggingface\transformers"

HF_HUB_CACHE_DIR = r"D:\huggingface\hub"
os.makedirs(HF_HUB_CACHE_DIR, exist_ok=True)

# ── Dataset ───────────────────────────────────────────────────────────────────
DATASET_FILE = os.path.join(os.path.dirname(__file__), "generated_answers_cleaned.csv")

if not os.path.exists(DATASET_FILE):
    raise FileNotFoundError(
        "\n" + "="*70 + "\n"
        "ERROR: Dataset no encontrado!\n\n"
        f"Archivo esperado: {DATASET_FILE}\n"
        + "="*70
    )

print(f"Cargando dataset: {DATASET_FILE}")
df = pd.read_csv(DATASET_FILE, encoding="latin1")

REQUIREMENT_COL  = "requirement"   # el intent / prompt
GROUND_TRUTH_COL = "model_answer"  # configuracion de referencia (ground truth)

assert REQUIREMENT_COL  in df.columns, f"Columna '{REQUIREMENT_COL}' no encontrada"
assert GROUND_TRUTH_COL in df.columns, f"Columna '{GROUND_TRUTH_COL}' no encontrada"

print(f"Muestras cargadas: {len(df)}")

# ── Modelos (mismos del PDF, mismos paths que clasificacion) ──────────────────
MODELS = {
    "Zephyr-7B": {
        "path":   "HuggingFaceH4/zephyr-7b-beta",
        "params": "7B",
    },
    "Llama-3.2-3B-Instruct": {
        "path":   "meta-llama/Llama-3.2-3B-Instruct",
        "params": "3B",
    },
    "Phi-3.5-mini-instruct": {
        "path":   "microsoft/Phi-3.5-mini-instruct",
        "params": "3.8B",
    },
    "Qwen2.5-3B-Instruct": {
        "path":   "Qwen/Qwen2.5-3B-Instruct",
        "params": "3B",
    },
    "Qwen2.5-7B-Instruct": {
        "path":   "Qwen/Qwen2.5-7B-Instruct",
        "params": "7B",
    },
    "Gemma-2-9B-it": {
        "path":   "google/gemma-2-9b-it",
        "params": "9B",
    },
    "Qwen2.5-14B-Instruct": {
        "path":   "Qwen/Qwen2.5-14B-Instruct",
        "params": "14B",
    },
    "Phi-4": {
        "path":   "microsoft/phi-4",
        "params": "14B",
    },
}

# ── Prompt ────────────────────────────────────────────────────────────────────
GENERATION_PROMPT = """You are a Cisco network engineer working on an Intent-Based Networking (IBN) system.
Given the following network configuration requirement, generate the exact Cisco IOS configuration commands needed.

Rules:
- Output ONLY the configuration commands, nothing else.
- Do not include explanations, comments, or markdown headers.
- Use standard Cisco IOS syntax.
- If the requirement involves multiple devices, separate each device block clearly.

Requirement: {requirement}

Configuration:"""


# ── Carga del modelo (mismo patron que clasificacion) ─────────────────────────
def load_model(model_name, model_config):
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM

        if not torch.cuda.is_available():
            raise RuntimeError(
                "\nERROR: CUDA no disponible. Verifica que:\n"
                "  1. PyTorch con CUDA esta instalado\n"
                "  2. Los drivers de NVIDIA estan actualizados\n"
                "  3. Ejecuta: nvidia-smi para verificar la GPU"
            )

        device = "cuda:0"
        print(f"\nCargando {model_name} en {torch.cuda.get_device_name(0)}...")
        print(f"  Cache HF: {HF_HUB_CACHE_DIR}")

        tokenizer = AutoTokenizer.from_pretrained(
            model_config["path"],
            trust_remote_code=model_config.get("trust_remote_code", False),
            cache_dir=HF_HUB_CACHE_DIR,
            token=HF_TOKEN,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_config["path"],
            dtype=torch.bfloat16,
            device_map=device,
            trust_remote_code=model_config.get("trust_remote_code", False),
            cache_dir=HF_HUB_CACHE_DIR,
            token=HF_TOKEN,
        )

        print(f"  VRAM usada: {torch.cuda.memory_allocated(0)/1024**3:.2f} GB")
        print(f"  Modelo {model_name} cargado exitosamente")
        return tokenizer, model

    except Exception as e:
        print(f"Error cargando {model_name}: {e}")
        return None, None


# ── Inferencia: genera una configuracion ─────────────────────────────────────
def generate_config(requirement, tokenizer, model, max_new_tokens=512):
    try:
        import torch

        messages = [
            {
                "role": "user",
                "content": GENERATION_PROMPT.format(requirement=requirement),
            }
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=2048
        ).to("cuda:0")

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                temperature=None,
                top_p=None,
            )

        generated = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        ).strip()

        return generated

    except Exception as e:
        print(f"  Error en generacion: {e}")
        return "ERROR"


# ── Metricas ROUGE ────────────────────────────────────────────────────────────
def compute_rouge(predictions, references):
    from rouge_score import rouge_scorer as rs

    scorer = rs.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    r1, r2, rl = [], [], []

    for pred, ref in zip(predictions, references):
        if not pred or not ref or pred == "ERROR":
            continue
        s = scorer.score(ref, pred)
        r1.append(s["rouge1"].fmeasure)
        r2.append(s["rouge2"].fmeasure)
        rl.append(s["rougeL"].fmeasure)

    return {
        "rouge1":      round(float(np.mean(r1)),  4) if r1 else 0.0,
        "rouge2":      round(float(np.mean(r2)),  4) if r2 else 0.0,
        "rougeL":      round(float(np.mean(rl)),  4) if rl else 0.0,
        "rouge1_std":  round(float(np.std(r1)),   4) if r1 else 0.0,
        "rouge2_std":  round(float(np.std(r2)),   4) if r2 else 0.0,
        "rougeL_std":  round(float(np.std(rl)),   4) if rl else 0.0,
    }


# ── Metricas BERTScore ────────────────────────────────────────────────────────
def compute_bertscore(predictions, references):
    from bert_score import score as bscore

    valid_pairs = [
        (p, r) for p, r in zip(predictions, references)
        if p and r and p != "ERROR"
    ]

    if not valid_pairs:
        return {
            "bertscore_p": 0.0, "bertscore_r": 0.0,
            "bertscore_f1": 0.0, "bertscore_f1_std": 0.0,
        }

    preds, refs = zip(*valid_pairs)

    # Algunas versiones de bert-score no incluyen codebert en model2layers.
    # Probamos con CodeBERT (num_layers explicito) y luego fallback.
    candidates = [
        ("microsoft/codebert-base", 12),
        ("roberta-large", None),
    ]

    last_error = None
    P = R = F1 = None

    for model_name, num_layers in candidates:
        try:
            kwargs = {
                "lang": "en",
                "model_type": model_name,
                "verbose": False,
                "batch_size": 16,
            }
            if num_layers is not None:
                kwargs["num_layers"] = num_layers

            P, R, F1 = bscore(list(preds), list(refs), **kwargs)
            print(f"    BERTScore usando: {model_name}")
            break
        except Exception as e:
            last_error = e
            print(f"    Aviso BERTScore con {model_name} fallo: {e}")

    if F1 is None:
        print(f"    Error: no se pudo calcular BERTScore. Ultimo error: {last_error}")
        return {
            "bertscore_p": 0.0, "bertscore_r": 0.0,
            "bertscore_f1": 0.0, "bertscore_f1_std": 0.0,
        }

    return {
        "bertscore_p":      round(float(P.mean()),  4),
        "bertscore_r":      round(float(R.mean()),  4),
        "bertscore_f1":     round(float(F1.mean()), 4),
        "bertscore_f1_std": round(float(F1.std()),  4),
    }


# ── Evaluacion de un modelo ───────────────────────────────────────────────────
def evaluate_model(model_name, model_config, df_eval):
    import torch

    print(f"\n{'='*80}")
    print(f"EVALUANDO: {model_name} ({model_config['params']})")
    print(f"{'='*80}")

    tokenizer, model = load_model(model_name, model_config)
    if tokenizer is None or model is None:
        print(f"Saltando {model_name} — no se pudo cargar.")
        return None
    try:
        predictions = []
        latencies   = []
        n = len(df_eval)

        for i, (_, row) in enumerate(df_eval.iterrows()):
            if i % 10 == 0:
                print(f"  Procesando {i + 1}/{n}...")

            t0      = time.time()
            pred    = generate_config(row[REQUIREMENT_COL], tokenizer, model)
            elapsed = time.time() - t0

            predictions.append(pred)
            latencies.append(elapsed)

        references  = df_eval[GROUND_TRUTH_COL].tolist()
        error_count = predictions.count("ERROR")

        print(f"\n  Calculando ROUGE...")
        rouge_metrics = compute_rouge(predictions, references)

        print(f"  Calculando BERTScore (codebert-base)...")
        bert_metrics = compute_bertscore(predictions, references)

        total_time = sum(latencies)
        avg_time   = float(np.mean(latencies))

        results = {
            "model_name":          model_name,
            "params":              model_config["params"],
            "samples_evaluated":   n,
            "error_count":         int(error_count),
            "error_rate":          round(error_count / n, 4),
            "total_time_s":        round(total_time, 2),
            "avg_time_per_sample": round(avg_time, 4),
            **rouge_metrics,
            **bert_metrics,
            "predictions":         predictions,   # guardadas para analisis posterior
        }

        print(f"\n  Resultados {model_name}:")
        print(f"    ROUGE-1:        {rouge_metrics['rouge1']:.4f}  (std {rouge_metrics['rouge1_std']:.4f})")
        print(f"    ROUGE-2:        {rouge_metrics['rouge2']:.4f}  (std {rouge_metrics['rouge2_std']:.4f})")
        print(f"    ROUGE-L:        {rouge_metrics['rougeL']:.4f}  (std {rouge_metrics['rougeL_std']:.4f})")
        print(f"    BERTScore-F1:   {bert_metrics['bertscore_f1']:.4f}  (std {bert_metrics['bertscore_f1_std']:.4f})")
        print(f"    Tiempo/muestra: {avg_time:.3f}s")
        print(f"    Errores:        {error_count}/{n}")

        return results
    finally:
        # Forzamos liberar referencias pesadas antes de cargar el siguiente modelo.
        del model
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            print(
                "  VRAM post-modelo: "
                f"allocada {torch.cuda.memory_allocated(0)/1024**3:.2f} GB | "
                f"reservada {torch.cuda.memory_reserved(0)/1024**3:.2f} GB"
            )


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    import torch

    print("=" * 80)
    print("EVALUACION DE GENERACION DE CONFIGURACIONES CISCO")
    print(f"Inicio: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    print(f"\nDataset: {len(df)} muestras")
    print(f"Modelos a evaluar: {len(MODELS)}\n")

    all_results = []

    for model_name, model_config in MODELS.items():
        result = evaluate_model(model_name, model_config, df)
        if result:
            all_results.append(result)

    # ── Guardar resultados ────────────────────────────────────────────────────
    timestamp    = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"generation_results_{timestamp}.json"

    output = {
        "timestamp":        timestamp,
        "dataset":          DATASET_FILE,
        "total_samples":    len(df),
        "models_evaluated": list(MODELS.keys()),
        "results":          all_results,
    }

    with open(results_file, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n{'='*80}")
    print(f"RESULTADOS GUARDADOS EN: {results_file}")
    print(f"{'='*80}")

    # ── Tabla comparativa final ───────────────────────────────────────────────
    header = f"{'Modelo':<30} {'ROUGE-1':<10} {'ROUGE-2':<10} {'ROUGE-L':<10} {'BERT-F1':<12} {'T/muestra'}"
    print(f"\n{'COMPARACION FINAL':^80}")
    print(header)
    print("-" * 80)

    for r in sorted(all_results, key=lambda x: x["bertscore_f1"], reverse=True):
        print(
            f"{r['model_name']:<30} "
            f"{r['rouge1']:<10.4f} "
            f"{r['rouge2']:<10.4f} "
            f"{r['rougeL']:<10.4f} "
            f"{r['bertscore_f1']:<12.4f} "
            f"{r['avg_time_per_sample']:.3f}s"
        )


if __name__ == "__main__":
    main()

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

# ── Entorno HuggingFace ───────────────────────────────────────────────────────
ENV_FILE = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=ENV_FILE)

HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError(f"No se encontro HF_TOKEN en {ENV_FILE}")

os.environ["HF_TOKEN"] = HF_TOKEN
os.environ["HF_HOME"] = r"D:\huggingface"
os.environ["HUGGINGFACE_HUB_CACHE"] = r"D:\huggingface\hub"
os.environ["TRANSFORMERS_CACHE"] = r"D:\huggingface\transformers"

HF_HUB_CACHE_DIR = r"D:\huggingface\hub"
os.makedirs(HF_HUB_CACHE_DIR, exist_ok=True)

# ── Dataset ───────────────────────────────────────────────────────────────────
DATASET_FILE = os.path.join(os.path.dirname(__file__), "dataset_v2.csv")

if not os.path.exists(DATASET_FILE):
    raise FileNotFoundError(
        "\n" + "=" * 70 + "\n"
        "ERROR: Dataset no encontrado!\n\n"
        f"Archivo esperado: {DATASET_FILE}\n"
        + "=" * 70
    )

print(f"Cargando dataset: {DATASET_FILE}")
try:
    df = pd.read_csv(DATASET_FILE, encoding="utf-8")
except UnicodeDecodeError:
    df = pd.read_csv(DATASET_FILE, encoding="latin1")

REQUIREMENT_COL = "requirement"
GROUND_TRUTH_COL = "ground_truth"

assert REQUIREMENT_COL in df.columns, f"Columna '{REQUIREMENT_COL}' no encontrada"
assert GROUND_TRUTH_COL in df.columns, f"Columna '{GROUND_TRUTH_COL}' no encontrada"

print(f"Muestras cargadas: {len(df)}")

# ── Solo Qwen ────────────────────────────────────────────────────────────────
MODEL_NAME = "Qwen2.5-7B"
MODEL_CONFIG = {
    "path": "Qwen/Qwen2.5-7B-Instruct",
    "params": "7B",
}
DEVICE = "cuda:0"

# False -> generacion directa
# True  -> pipeline con plan (2 pasos)
USE_PLANNING = True

# ── Prompt ────────────────────────────────────────────────────────────────────
NETWORK_CONTEXT = """=== NETWORK TOPOLOGY ===

Interface table (each row is one physical link):

DEVICE  INTERFACE   IP ADDRESS      MASK             NEIGHBOR  NEIGHBOR INTERFACE
R1      Ethernet0/0 10.0.1.1        255.255.255.0    SW1       Fa0/24
R1      Ethernet0/1 10.0.12.1       255.255.255.0    R2        Ethernet0/0
R1      Ethernet0/2 10.0.14.1       255.255.255.0    R4        Ethernet0/0
R2      Ethernet0/0 10.0.12.2       255.255.255.0    R1        Ethernet0/1
R2      Ethernet0/1 10.0.23.1       255.255.255.0    R3        Ethernet0/0
R2      Ethernet0/2 10.0.24.1       255.255.255.0    R4        Ethernet0/1
R3      Ethernet0/0 10.0.23.2       255.255.255.0    R2        Ethernet0/1
R3      Ethernet0/1 10.0.34.1       255.255.255.0    R4        Ethernet0/2
R3      Ethernet0/2 10.0.2.1        255.255.255.0    SW2       Fa0/24
R4      Ethernet0/0 10.0.14.2       255.255.255.0    R1        Ethernet0/2
R4      Ethernet0/1 10.0.24.2       255.255.255.0    R2        Ethernet0/2
R4      Ethernet0/2 10.0.34.2       255.255.255.0    R3        Ethernet0/1

Hosts:
- h1: IP 10.0.1.10, mask 255.255.255.0, gateway 10.0.1.1, connected to SW1 Fa0/1
- h2: IP 10.0.2.10, mask 255.255.255.0, gateway 10.0.2.1, connected to SW2 Fa0/1

Switch ports:
- SW1 Fa0/1  : access port, VLAN 10, connected to h1
- SW1 Fa0/24 : trunk port, connected to R1 Ethernet0/0, VLANs allowed: 10, 20
- SW2 Fa0/1  : access port, VLAN 30, connected to h2
- SW2 Fa0/24 : trunk port, connected to R3 Ethernet0/2, VLANs allowed: 30, 40
"""

GENERATION_PROMPT = """You are a Cisco network engineer.
Generate exact Cisco IOS configuration from the given topology.

=== RULES ===
- Output ONLY Cisco IOS commands. No explanations, comments, or markdown.
- Use ONLY interfaces, IPs, VLANs, and technologies defined in the topology.
- Apply configuration to the correct device.
- Each device block must start with `configure terminal` and end with `end`.
- When configuring multiple devices, output each block separately in order.
- Use wildcard masks for OSPF and ACLs. Use dotted-decimal subnet masks for `ip address`.
- Use next-hop IP addresses for static routes.
- If the requirement cannot be implemented with the given topology, output exactly: NO_CODE

=== EXAMPLES ===

Example 1 — Single device, interface configuration:
Requirement: Configure the IP address on R1's interface connecting to R4.
Configuration:
R1# configure terminal
R1(config)# interface Ethernet0/2
R1(config-if)# ip address 10.0.14.1 255.255.255.0
R1(config-if)# no shutdown
R1(config-if)# end

Example 2 — Multiple devices, symmetric configuration:
Requirement: Configure OSPF authentication on the link between R1 and R2 using password ospfkey.
Configuration:
R1# configure terminal
R1(config)# interface Ethernet0/1
R1(config-if)# ip ospf authentication
R1(config-if)# ip ospf authentication-key ospfkey
R1(config-if)# end

R2# configure terminal
R2(config)# interface Ethernet0/0
R2(config-if)# ip ospf authentication
R2(config-if)# ip ospf authentication-key ospfkey
R2(config-if)# end

Example 3 — Named ACL with explicit permit at the end:
Requirement: Configure an extended ACL on R1 to deny TCP traffic from 10.0.1.0/24 to any destination on port 23.
Configuration:
R1# configure terminal
R1(config)# ip access-list extended DENY_TELNET
R1(config-ext-nacl)# deny tcp 10.0.1.0 0.0.0.255 any eq 23
R1(config-ext-nacl)# permit ip any any
R1(config-ext-nacl)# end

=== INPUT ===

Topology:
{network_context}

Requirement:
{requirement}

Configuration:
"""

PLANNING_PROMPT = """You are a senior network engineer.

Identify the key points required to generate a correct Cisco IOS configuration.

Rules:

* Focus on key configuration points
* Output a concise bullet list using "- " at the start of each line
* Do NOT include Cisco IOS commands
* Include only points that are necessary and relevant to satisfy the requirement
* Cover critical dimensions when applicable: devices, interfaces, addressing, protocols, policy/ACL order, dependencies, and validation checks
* Do not invent values, devices, links, IPs, VLANs, or constraints outside the topology

Output format example:
- Key point one
- Key point two
- Key point three
- ...

Do not output explanations, comments, or markdown.

Topology:
{network_context}

Requirement:
{requirement}

Plan:
"""

GENERATION_WITH_PLAN_PROMPT = """You are a Cisco network engineer.
Generate exact Cisco IOS configuration from the given plan and topology.

=== RULES ===
- Output ONLY Cisco IOS commands. No explanations, comments, or markdown.
- Use the plan as guidance for key points when generating the configuration.
- Use ONLY interfaces, IPs, VLANs, and technologies defined in the topology.
- Apply configuration to the correct device.
- Each device block must start with `configure terminal` and end with `end`.
- When configuring multiple devices, output each block separately in order.
- Use wildcard masks for OSPF and ACLs. Use dotted-decimal subnet masks for `ip address`.
- Use next-hop IP addresses for static routes.
- If the requirement cannot be implemented with the given topology, output exactly: NO_CODE

=== EXAMPLES ===

Example 1 — Single device, interface configuration:
Requirement: Configure the IP address on R1's interface connecting to R4.
Configuration:
R1# configure terminal
R1(config)# interface Ethernet0/2
R1(config-if)# ip address 10.0.14.1 255.255.255.0
R1(config-if)# no shutdown
R1(config-if)# end

Example 2 — Multiple devices, symmetric configuration:
Requirement: Configure OSPF authentication on the link between R1 and R2 using password ospfkey.
Configuration:
R1# configure terminal
R1(config)# interface Ethernet0/1
R1(config-if)# ip ospf authentication
R1(config-if)# ip ospf authentication-key ospfkey
R1(config-if)# end

R2# configure terminal
R2(config)# interface Ethernet0/0
R2(config-if)# ip ospf authentication
R2(config-if)# ip ospf authentication-key ospfkey
R2(config-if)# end

Example 3 — Named ACL with explicit permit at the end:
Requirement: Configure an extended ACL on R1 to deny TCP traffic from 10.0.1.0/24 to any destination on port 23.
Configuration:
R1# configure terminal
R1(config)# ip access-list extended DENY_TELNET
R1(config-ext-nacl)# deny tcp 10.0.1.0 0.0.0.255 any eq 23
R1(config-ext-nacl)# permit ip any any
R1(config-ext-nacl)# end

=== INPUT ===

Plan:
{plan}

Topology:
{network_context}

Requirement:
{requirement}

Configuration:
"""


def load_model(model_config):
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

        # CAMBIO 5: Seleccionar dtype segun soporte de la GPU
        dtype = (
            torch.bfloat16
            if torch.cuda.is_bf16_supported()
            else torch.float16
        )

        print(f"\nCargando {MODEL_NAME} en {torch.cuda.get_device_name(0)}...")
        print(f"  dtype seleccionado: {dtype}")
        print(f"  Cache HF: {HF_HUB_CACHE_DIR}")

        tokenizer = AutoTokenizer.from_pretrained(
            model_config["path"],
            cache_dir=HF_HUB_CACHE_DIR,
            token=HF_TOKEN,
        )

       
        model = AutoModelForCausalLM.from_pretrained(
            model_config["path"],
            dtype=dtype,      # ← corregido
            device_map=DEVICE,
            cache_dir=HF_HUB_CACHE_DIR,
            token=HF_TOKEN,
        )

        print(f"  VRAM usada: {torch.cuda.memory_allocated(0)/1024**3:.2f} GB")
        print(f"  Modelo {MODEL_NAME} cargado exitosamente")
        return tokenizer, model

    except Exception as e:
        print(f"Error cargando {MODEL_NAME}: {e}")
        return None, None


def generate_config(requirement, tokenizer, model, max_new_tokens=512):
    try:
        import torch

        messages = [
            {
                "role": "user",
                "content": GENERATION_PROMPT.format(
                    requirement=requirement,
                    network_context=NETWORK_CONTEXT,
                ),
            }
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # CAMBIO 2: .to(DEVICE) -> .to(model.device)
        inputs = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=2048
        ).to(model.device)

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


def generate_with_plan(requirement, tokenizer, model):
    try:
        import torch
        import re

        planning_prompt = PLANNING_PROMPT.format(
            requirement=requirement,
            network_context=NETWORK_CONTEXT,
        )
        plan_messages = [{"role": "user", "content": planning_prompt}]
        plan_chat_prompt = tokenizer.apply_chat_template(
            plan_messages, tokenize=False, add_generation_prompt=True
        )

        plan_inputs = tokenizer(
            plan_chat_prompt, return_tensors="pt", truncation=True, max_length=2048
        ).to(model.device)

        with torch.no_grad():
            plan_outputs = model.generate(
                **plan_inputs,
                max_new_tokens=128,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                temperature=None,
                top_p=None,
            )

        plan = tokenizer.decode(
            plan_outputs[0][plan_inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        ).strip()

        generation_prompt = GENERATION_WITH_PLAN_PROMPT.format(
            plan=plan,
            requirement=requirement,
            network_context=NETWORK_CONTEXT,
        )
        cfg_messages = [{"role": "user", "content": generation_prompt}]
        cfg_chat_prompt = tokenizer.apply_chat_template(
            cfg_messages, tokenize=False, add_generation_prompt=True
        )

        cfg_inputs = tokenizer(
            cfg_chat_prompt, return_tensors="pt", truncation=True, max_length=2048
        ).to(model.device)

        with torch.no_grad():
            cfg_outputs = model.generate(
                **cfg_inputs,
                max_new_tokens=256,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                temperature=None,
                top_p=None,
            )

        generated = tokenizer.decode(
            cfg_outputs[0][cfg_inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        ).strip()

        stop_markers = [
            "Plan:", "Topology:", "Requirement:", "Configuration:",
            "Output:", "Explanation:", "Note:", "===",
        ]
        cut_positions = [generated.find(m) for m in stop_markers if generated.find(m) != -1]
        if cut_positions:
            generated = generated[:min(cut_positions)].strip()

        if re.search(r"\bNO_CODE\b", generated, re.IGNORECASE):
            has_ios = bool(re.search(r"\b(config|interface|router|ip\s|switchport|access-list|route-map|vlan|spanning-tree|line\s+vty|hostname|enable|copy\s+running-config|end)\b", generated, re.IGNORECASE))
            if not has_ios:
                generate_with_plan.last_plan = plan
                return "NO_CODE"
            generated = re.sub(r"\bOR\b\s*\bNO_CODE\b.*$", "", generated, flags=re.IGNORECASE | re.DOTALL).strip()

        if not generated:
            generate_with_plan.last_plan = plan
            return "NO_CODE"

        generate_with_plan.last_plan = plan
        return generated

    except Exception as e:
        print(f"  Error en pipeline con plan: {e}")
        generate_with_plan.last_plan = ""
        return "NO_CODE"


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
        "rouge1": round(float(np.mean(r1)), 4) if r1 else 0.0,
        "rouge2": round(float(np.mean(r2)), 4) if r2 else 0.0,
        "rougeL": round(float(np.mean(rl)), 4) if rl else 0.0,
        "rouge1_std": round(float(np.std(r1)), 4) if r1 else 0.0,
        "rouge2_std": round(float(np.std(r2)), 4) if r2 else 0.0,
        "rougeL_std": round(float(np.std(rl)), 4) if rl else 0.0,
    }


def compute_bertscore(predictions, references):
    from bert_score import score as bscore

    valid_pairs = [
        (p, r) for p, r in zip(predictions, references)
        if p and r and p != "ERROR"
    ]

    if not valid_pairs:
        return {
            "bertscore_p": 0.0,
            "bertscore_r": 0.0,
            "bertscore_f1": 0.0,
            "bertscore_f1_std": 0.0,
        }

    preds, refs = zip(*valid_pairs)
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
            "bertscore_p": 0.0,
            "bertscore_r": 0.0,
            "bertscore_f1": 0.0,
            "bertscore_f1_std": 0.0,
        }

    return {
        "bertscore_p": round(float(P.mean()), 4),
        "bertscore_r": round(float(R.mean()), 4),
        "bertscore_f1": round(float(F1.mean()), 4),
        "bertscore_f1_std": round(float(F1.std()), 4),
    }


def evaluate_qwen(df_eval):
    import torch

    print(f"\n{'=' * 80}")
    print(f"EVALUANDO: {MODEL_NAME} ({MODEL_CONFIG['params']})")
    print(f"Planning pipeline: {USE_PLANNING}")
    print(f"{'=' * 80}")

    tokenizer, model = load_model(MODEL_CONFIG)
    if tokenizer is None or model is None:
        print(f"Saltando {MODEL_NAME} - no se pudo cargar.")
        return None

    try:
        predictions = []
        plans = []
        latencies = []
        n = len(df_eval)

        for i, (_, row) in enumerate(df_eval.iterrows()):
            if i % 10 == 0:
                print(f"  Procesando {i + 1}/{n}...")

            t0 = time.time()
            if USE_PLANNING:
                pred = generate_with_plan(row[REQUIREMENT_COL], tokenizer, model)
                plans.append(getattr(generate_with_plan, "last_plan", ""))
            else:
                pred = generate_config(row[REQUIREMENT_COL], tokenizer, model)
                plans.append("")
            elapsed = time.time() - t0

            predictions.append(pred)
            latencies.append(elapsed)

        references = df_eval[GROUND_TRUTH_COL].tolist()
        error_count = predictions.count("ERROR")

        print("\n  Calculando ROUGE...")
        rouge_metrics = compute_rouge(predictions, references)

        print("  Calculando BERTScore (codebert-base)...")
        bert_metrics = compute_bertscore(predictions, references)

        total_time = sum(latencies)
        avg_time = float(np.mean(latencies))

        results = {
            "model_name": MODEL_NAME,
            "params": MODEL_CONFIG["params"],
            "samples_evaluated": n,
            "error_count": int(error_count),
            "error_rate": round(error_count / n, 4),
            "total_time_s": round(total_time, 2),
            "avg_time_per_sample": round(avg_time, 4),
            **rouge_metrics,
            **bert_metrics,
            "plans": plans,
            "predictions": predictions,
        }

        print(f"\n  Resultados {MODEL_NAME}:")
        print(f"    ROUGE-1:        {rouge_metrics['rouge1']:.4f}  (std {rouge_metrics['rouge1_std']:.4f})")
        print(f"    ROUGE-2:        {rouge_metrics['rouge2']:.4f}  (std {rouge_metrics['rouge2_std']:.4f})")
        print(f"    ROUGE-L:        {rouge_metrics['rougeL']:.4f}  (std {rouge_metrics['rougeL_std']:.4f})")
        print(f"    BERTScore-F1:   {bert_metrics['bertscore_f1']:.4f}  (std {bert_metrics['bertscore_f1_std']:.4f})")
        print(f"    Tiempo/muestra: {avg_time:.3f}s")
        print(f"    Errores:        {error_count}/{n}")

        return results

    finally:
        del model
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            # CAMBIO 4: eliminado ipc_collect() innecesario en proceso unico
            print(
                "  VRAM post-modelo: "
                f"allocada {torch.cuda.memory_allocated(0)/1024**3:.2f} GB | "
                f"reservada {torch.cuda.memory_reserved(0)/1024**3:.2f} GB"
            )


def main():
    print("=" * 80)
    print("EVALUACION DE GENERACION DE CONFIGURACIONES CISCO - SOLO QWEN")
    print(f"Inicio: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    print(f"\nDataset: {len(df)} muestras")

    result = evaluate_qwen(df)
    if not result:
        print("No se obtuvieron resultados.")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode = "con_plan" if USE_PLANNING else "sin_plan"
    results_file = f"generation_results_qwen_{mode}_{timestamp}.json"

    output = {
        "timestamp": timestamp,
        "dataset": DATASET_FILE,
        "total_samples": len(df),
        "models_evaluated": [MODEL_NAME],
        "use_planning": USE_PLANNING,
        "results": [result],
    }

    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 80}")
    print(f"RESULTADOS GUARDADOS EN: {results_file}")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
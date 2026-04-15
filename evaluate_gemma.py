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


def get_quantization_config(model_config):
    ""
    import torch
    from transformers import BitsAndBytesConfig

    quant = model_config.get("quantization", None)

    if quant == "int8":
        return BitsAndBytesConfig(load_in_8bit=True)
    elif quant == "int4":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    return None  # Sin cuantizacion, usa dtype del modelo


def get_quantization_label(model_config):
    quant = model_config.get("quantization", None)
    return quant if quant else "none"

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
DATASET_FILE = os.path.join(os.path.dirname(__file__), "dataset_v2.csv")

if not os.path.exists(DATASET_FILE):
    raise FileNotFoundError(
        "\n" + "="*70 + "\n"
        "ERROR: Dataset no encontrado!\n\n"
        f"Archivo esperado: {DATASET_FILE}\n"
        + "="*70
    )

print(f"Cargando dataset: {DATASET_FILE}")
try:
    df = pd.read_csv(DATASET_FILE, encoding="utf-8")
except UnicodeDecodeError:
    df = pd.read_csv(DATASET_FILE, encoding="latin1")

REQUIREMENT_COL  = "requirement"   # el intent / prompt
GROUND_TRUTH_COL = "ground_truth"  # configuracion de referencia (ground truth)

assert REQUIREMENT_COL  in df.columns, f"Columna '{REQUIREMENT_COL}' no encontrada"
assert GROUND_TRUTH_COL in df.columns, f"Columna '{GROUND_TRUTH_COL}' no encontrada"

print(f"Muestras cargadas: {len(df)}")

# ── Modelos (mismos del PDF, mismos paths que clasificacion) ──────────────────
MODELS = {
 
   'Llama-3.1-8B-Instruct': {
        'path': 'Qwen/Qwen2.5-7B-Instruct',
        'params': '8B',
        'quantization': 'none',
    },
}

USE_PLANNING = False

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

GENERATION_PROMPT = """You are a Cisco network engineer. Generate exact Cisco IOS configuration from the topology and requirement.

=== RULES ===
- Output ONLY Cisco IOS commands. No explanations, comments, or markdown.
- Use ONLY interfaces, IPs, VLANs, and technologies defined in the topology.
- Apply configuration to the correct device.
- Each device block must start with `configure terminal` and end with `end`.
- When configuring multiple devices, output each device block separately in order.
- Use wildcard masks for OSPF and ACLs. Use dotted-decimal subnet masks for `ip address`.
- Use next-hop IP addresses for static routes.
- If the requirement cannot be implemented with the given topology, output exactly: NO_CODE

=== EXAMPLES ===

Example 1 — Single device, interface configuration:
Requirement: Set the description 'Link to SW1' on R1's interface connecting to SW1.
Configuration:
R1# configure terminal
R1(config)# interface Ethernet0/0
R1(config-if)# description Link to SW1
R1(config-if)# end

Example 2 — Single device, routing:
Requirement: Configure a static route on R4 to reach the 10.0.23.0/24 network via R2.
Configuration:
R4# configure terminal
R4(config)# ip route 10.0.23.0 255.255.255.0 10.0.24.1
R4(config)# end

Example 3 — Single device, named ACL with explicit permit at the end:
Requirement: Configure an extended ACL on R2 to deny UDP traffic from 10.0.12.0/24 to any destination on port 161.
Configuration:
R2# configure terminal
R2(config)# ip access-list extended DENY_SNMP
R2(config-ext-nacl)# deny udp 10.0.12.0 0.0.0.255 any eq 161
R2(config-ext-nacl)# permit ip any any
R2(config-ext-nacl)# end

Example 4 — Multiple devices, symmetric configuration:
Requirement: Set the MTU to 1400 on both ends of the link between R1 and R2.
Configuration:
R1# configure terminal
R1(config)# interface Ethernet0/1
R1(config-if)# mtu 1400
R1(config-if)# end

R2# configure terminal
R2(config)# interface Ethernet0/0
R2(config-if)# mtu 1400
R2(config-if)# end

Example 5 — Multiple devices, same command on each:
Requirement: Configure NTP server 10.0.1.1 on R1, R2, and R4.
Configuration:
R1# configure terminal
R1(config)# ntp server 10.0.1.1
R1(config)# end

R2# configure terminal
R2(config)# ntp server 10.0.1.1
R2(config)# end

R4# configure terminal
R4(config)# ntp server 10.0.1.1
R4(config)# end

=== INPUT ===

Topology:
{network_context}

Requirement:
{requirement}

Configuration:"""


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


def build_prompt(tokenizer, messages, plain_prompt):
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    return plain_prompt


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
        quant_label = get_quantization_label(model_config)

        print(f"\nCargando {model_name} en {torch.cuda.get_device_name(0)}...")
        print(f"  Cuantizacion: {quant_label}")
        print(f"  Cache HF: {HF_HUB_CACHE_DIR}")

        quant_config = get_quantization_config(model_config)

        tokenizer = AutoTokenizer.from_pretrained(
            model_config["path"],
            trust_remote_code=model_config.get("trust_remote_code", False),
            cache_dir=HF_HUB_CACHE_DIR,
            token=HF_TOKEN,
        )

        load_kwargs = {
            "device_map": device,
            "cache_dir": HF_HUB_CACHE_DIR,
            "token": HF_TOKEN,
        }

        if quant_config:
            load_kwargs["quantization_config"] = quant_config
        else:
            load_kwargs["torch_dtype"] = torch.bfloat16

        model = AutoModelForCausalLM.from_pretrained(
            model_config["path"],
            **load_kwargs,
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

        plain_prompt = GENERATION_PROMPT.format(
            requirement=requirement,
            network_context=NETWORK_CONTEXT,
        )
        messages = [
            {
                "role": "user",
                "content": plain_prompt,
            }
        ]
        prompt = build_prompt(tokenizer, messages, plain_prompt)
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
                top_k=None,
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

        # Step 1: generate high-level plan
        planning_prompt = PLANNING_PROMPT.format(
            requirement=requirement,
            network_context=NETWORK_CONTEXT,
        )

        plan_messages = [{"role": "user", "content": planning_prompt}]
        plan_chat_prompt = build_prompt(tokenizer, plan_messages, planning_prompt)
        plan_inputs = tokenizer(
            plan_chat_prompt, return_tensors="pt", truncation=True, max_length=2048
        ).to("cuda:0")
        with torch.no_grad():
            plan_outputs = model.generate(
                **plan_inputs,
                max_new_tokens=128,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                temperature=None,
                top_p=None,
                top_k=None,
            )
        plan = tokenizer.decode(
            plan_outputs[0][plan_inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        ).strip()

        # Step 2: generate final config using the plan
        generation_prompt = GENERATION_WITH_PLAN_PROMPT.format(
            plan=plan,
            requirement=requirement,
            network_context=NETWORK_CONTEXT,
        )

        cfg_messages = [{"role": "user", "content": generation_prompt}]
        cfg_chat_prompt = build_prompt(tokenizer, cfg_messages, generation_prompt)
        cfg_inputs = tokenizer(
            cfg_chat_prompt, return_tensors="pt", truncation=True, max_length=2048
        ).to("cuda:0")
        with torch.no_grad():
            cfg_outputs = model.generate(
                **cfg_inputs,
                max_new_tokens=256,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                temperature=None,
                top_p=None,
                top_k=None,
            )
        generated = tokenizer.decode(
            cfg_outputs[0][cfg_inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        ).strip()

        # Enforce final-output cleanliness: no plan/prose, only config or NO_CODE.
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

    quant_label = get_quantization_label(model_config)

    print(f"\n{'='*80}")
    print(f"EVALUANDO: {model_name} ({model_config['params']}) | cuantizacion={quant_label}")
    print(f"{'='*80}")

    tokenizer, model = load_model(model_name, model_config)
    generation_fn = generate_config

    if tokenizer is None or model is None:
        print(f"Saltando {model_name} — no se pudo cargar.")
        return None
    try:
        predictions = []
        plans = []
        latencies   = []
        n = len(df_eval)

        for i, (_, row) in enumerate(df_eval.iterrows()):
            if i % 10 == 0:
                print(f"  Procesando {i + 1}/{n}...")

            t0      = time.time()
            if USE_PLANNING:
                pred = generate_with_plan(row[REQUIREMENT_COL], tokenizer, model)
                plans.append(getattr(generate_with_plan, "last_plan", ""))
            else:
                pred = generation_fn(row[REQUIREMENT_COL], tokenizer, model)
                plans.append("")
            elapsed = time.time() - t0

            predictions.append(pred)
            latencies.append(elapsed)

        references  = df_eval[GROUND_TRUTH_COL].tolist()
        error_count = predictions.count("ERROR")

        print("\n  Calculando ROUGE...")
        rouge_metrics = compute_rouge(predictions, references)

        print("  Calculando BERTScore (codebert-base)...")
        bert_metrics = compute_bertscore(predictions, references)

        total_time = sum(latencies)
        avg_time   = float(np.mean(latencies))

        results = {
            "model_name":          model_name,
            "params":              model_config["params"],
            "quantization":        quant_label,
            "samples_evaluated":   n,
            "error_count":         int(error_count),
            "error_rate":          round(error_count / n, 4),
            "total_time_s":        round(total_time, 2),
            "avg_time_per_sample": round(avg_time, 4),
            **rouge_metrics,
            **bert_metrics,
            "plans":               plans,
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
    print(f"Modo de evaluacion: {'con plan' if USE_PLANNING else 'sin plan'}")
    print(f"Inicio: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    print(f"\nDataset: {len(df)} muestras")
    print(f"Modelos a evaluar: {len(MODELS)}\n")

    quantization_by_model = {
        model_name: get_quantization_label(model_config)
        for model_name, model_config in MODELS.items()
    }
    quantization_tags = sorted(set(quantization_by_model.values()))
    quantization_suffix = "_q-" + "-".join(quantization_tags)
    print(f"Cuantizacion(es) configurada(s): {', '.join(quantization_tags)}")

    all_results = []

    for model_name, model_config in MODELS.items():
        result = evaluate_model(model_name, model_config, df)
        if result:
            all_results.append(result)

    # ── Guardar resultados ────────────────────────────────────────────────────
    timestamp    = datetime.now().strftime("%Y%m%d_%H%M%S")
    planning_suffix = "_con_plan" if USE_PLANNING else "_sin_plan"
    results_file = f"generation_results{planning_suffix}{quantization_suffix}_{timestamp}.json"

    output = {
        "timestamp":        timestamp,
        "dataset":          DATASET_FILE,
        "total_samples":    len(df),
        "models_evaluated": list(MODELS.keys()),
        "quantization_by_model": quantization_by_model,
        "results":          all_results,
    }

    with open(results_file, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n{'='*80}")
    print(f"CUANTIZACION(ES) USADA(S): {', '.join(quantization_tags)}")
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

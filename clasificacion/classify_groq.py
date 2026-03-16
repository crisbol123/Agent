"""
Clasificador semántico del dataset SLM_netconfig
Usa Groq API con Llama 3.3 70B (gratuito)

Requisitos:
    pip install groq pandas tqdm

Uso:
    export GROQ_API_KEY="gsk_..."
    python classify_groq.py

    # Solo una muestra de prueba:
    python classify_groq.py --sample 50

    # Pasando la key directamente:
    python classify_groq.py --api-key gsk_...
"""

import os
import json
import time
import argparse
import pandas as pd
from pathlib import Path

try:
    from groq import Groq
except ImportError:
    raise SystemExit("Falta el SDK: pip install groq")

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        total = kwargs.get("total", "?")
        print(f"Procesando {total} registros...")
        return iterable


# ---------------------------------------------------------------------------
# Categorías y prompt
# ---------------------------------------------------------------------------

CATEGORIES = {
    "ROUTING": (
        "Dynamic routing protocols (OSPF, BGP, EIGRP, RIP), static routes, MPLS, VRF, "
        "multicast (PIM, IGMP, MSDP), route redistribution, first-hop redundancy (HSRP, VRRP, GLBP)."
    ),
    "SECURITY": (
        "Access control lists (ACLs), zone-based firewalls, IPsec, VPN tunnels (IKE, ISAKMP, "
        "crypto maps), AAA, RADIUS, PKI, certificates, protection against network attacks."
    ),
    "QOS": (
        "Quality of Service: traffic policies (policy-map, class-map), bandwidth shaping, "
        "policing, priority queuing, DSCP/IP precedence marking, voice and video optimization."
    ),
    "CONNECTIVITY": (
        "Physical and logical interfaces, IP addressing, NAT, VLANs, dot1q encapsulation, ARP, "
        "bridge domains, port-channels (LACP), overlay tunnels (GRE, VXLAN, DMVPN, NHRP), "
        "IPv6 transition mechanisms."
    ),
    "MONITORING": (
        "Network observability and management: IP SLA, SNMP, NetFlow, BFD, LLDP, NTP, "
        "syslog/logging, NBAR, packet capture, OAM (CFM, Y.1731), operation schedulers."
    ),
    "GENERAL": (
        "Conceptual, procedural, or platform-specific questions that do not clearly belong to "
        "a single technical domain. Includes CLI modes, default values, feature compatibility, "
        "or genuinely mixed topics."
    ),
}

SYSTEM_PROMPT = """You are a Cisco network expert and dataset annotator.
Classify the given network configuration question into exactly one category.

Categories:
{categories}

Rules:
- Choose the SINGLE most relevant category based on the question PRIMARY intent.
- If the question spans two domains, pick the one the question is MAINLY asking about.
- Use GENERAL only when the question genuinely does not fit any specific technical domain.
- Respond ONLY with valid JSON. No explanation, no markdown, no extra text before or after.

Response format (strict):
{{"category": "CATEGORY_NAME", "confidence": 0.95, "reason": "one short sentence"}}""".format(
    categories="\n".join(f"- {name}: {desc}" for name, desc in CATEGORIES.items())
)

CONFIDENCE_THRESHOLD = 0.7


# ---------------------------------------------------------------------------
# Clasificador
# ---------------------------------------------------------------------------

def classify_question(client: "Groq", question: str, retries: int = 4) -> dict:
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": question},
                ],
                max_tokens=150,
                temperature=0.0,
            )

            raw = response.choices[0].message.content.strip()

            # Limpiar backticks si el modelo los agrega
            if "```" in raw:
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            raw = raw.strip()

            result = json.loads(raw)

            # Validar categoría
            if result.get("category") not in CATEGORIES:
                return {
                    "category": "GENERAL",
                    "confidence": 0.0,
                    "reason": f"Invalid category returned: {result.get('category')}",
                }

            # Asegurar que confidence sea float entre 0 y 1
            result["confidence"] = float(result.get("confidence", 0.0))
            result["confidence"] = max(0.0, min(1.0, result["confidence"]))

            return result

        except json.JSONDecodeError as e:
            if attempt == retries - 1:
                return {
                    "category": "GENERAL",
                    "confidence": 0.0,
                    "reason": f"JSON parse error: {e} | raw: {raw[:80]}",
                }
            time.sleep(1)

        except Exception as e:
            err = str(e).lower()
            # Rate limit de Groq
            if "rate_limit" in err or "429" in err or "too many" in err:
                wait = 30 * (attempt + 1)
                print(f"\n  Rate limit — esperando {wait}s...")
                time.sleep(wait)
            elif attempt == retries - 1:
                return {
                    "category": "GENERAL",
                    "confidence": 0.0,
                    "reason": f"API error: {str(e)[:100]}",
                }
            else:
                time.sleep(3 * (attempt + 1))

    return {"category": "GENERAL", "confidence": 0.0, "reason": "Max retries exceeded"}


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------

def run(api_key: str, input_path: str, output_path: str,
        low_conf_path: str, checkpoint_path: str, sample: int | None):

    client = Groq(api_key=api_key)

    df = pd.read_csv(input_path)
    if sample:
        df = df.sample(n=sample, random_state=42).reset_index(drop=True)
        print(f"Modo prueba: {sample} registros aleatorios.")

    total = len(df)
    print(f"\nDataset: {total} preguntas")
    print(f"Modelo:  llama-3.3-70b-versatile (Groq)")
    print(f"Umbral de confianza: {CONFIDENCE_THRESHOLD}\n")

    # Cargar checkpoint
    checkpoint = Path(checkpoint_path)
    results: dict[int, dict] = {}
    if checkpoint.exists():
        with open(checkpoint) as f:
            results = {int(k): v for k, v in json.load(f).items()}
        print(f"Checkpoint encontrado: {len(results)} registros ya clasificados.")

    pending = [i for i in range(total) if i not in results]
    print(f"Pendientes: {len(pending)}\n")

    save_every = 100

    for count, idx in enumerate(tqdm(pending, total=len(pending), desc="Clasificando")):
        question = str(df.loc[idx, "question"])
        results[idx] = classify_question(client, question)

        # Groq free tier: ~30 req/min con llama-3.3-70b
        # 0.5s entre llamadas = ~120 req/min pero Groq throttlea, 
        # con 1s quedamos seguros en ~60 req/min
        time.sleep(1.0)

        if (count + 1) % save_every == 0:
            with open(checkpoint, "w") as f:
                json.dump(results, f)
            done = len(results)
            print(f"  [{done}/{total}] checkpoint guardado")

    # Guardar checkpoint final
    with open(checkpoint, "w") as f:
        json.dump(results, f)

    # Construir DataFrame final
    df["category"]   = [results[i]["category"]   for i in range(total)]
    df["confidence"] = [results[i]["confidence"] for i in range(total)]
    df["reason"]     = [results[i]["reason"]     for i in range(total)]

    # CSV principal (todo el dataset)
    df.to_csv(output_path, index=False)
    print(f"\nDataset completo guardado: {output_path}")

    # CSV de baja confianza (para revisión manual)
    low_conf = df[df["confidence"] < CONFIDENCE_THRESHOLD].copy()
    low_conf.to_csv(low_conf_path, index=False)
    print(f"Baja confianza (<{CONFIDENCE_THRESHOLD}): {len(low_conf)} registros → {low_conf_path}")

    # Resumen
    print("\n=== DISTRIBUCIÓN FINAL ===")
    stats   = df["category"].value_counts()
    pct     = df["category"].value_counts(normalize=True) * 100
    summary = pd.DataFrame({"N": stats, "%": pct.round(1)})
    print(summary)

    print(f"\nConfianza promedio: {df['confidence'].mean():.3f}")
    print(f"Registros confianza ≥ {CONFIDENCE_THRESHOLD}: {(df['confidence'] >= CONFIDENCE_THRESHOLD).sum()} "
          f"({(df['confidence'] >= CONFIDENCE_THRESHOLD).mean()*100:.1f}%)")
    print(f"Registros confianza <  {CONFIDENCE_THRESHOLD}: {len(low_conf)} "
          f"({len(low_conf)/total*100:.1f}%) → revisa manualmente antes de usar como ground truth")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clasificador SLM_netconfig con Groq + Llama 3.3 70B")
    parser.add_argument("--api-key",    type=str, default=os.getenv("GROQ_API_KEY"))
    parser.add_argument("--input",      type=str, default="requirements_questions_v2.csv")
    parser.add_argument("--output",     type=str, default="requirements_classified.csv",
                        help="CSV con todo el dataset clasificado")
    parser.add_argument("--low-conf",   type=str, default="requirements_low_confidence.csv",
                        help="CSV con registros de confianza < 0.7 para revisión manual")
    parser.add_argument("--checkpoint", type=str, default="groq_checkpoint.json")
    parser.add_argument("--sample",     type=int, default=None,
                        help="Procesar solo N registros (modo prueba)")
    args = parser.parse_args()

    if not args.api_key:
        raise SystemExit(
            "No se encontró GROQ_API_KEY.\n"
            "Opción 1: export GROQ_API_KEY='gsk_...'\n"
            "Opción 2: python classify_groq.py --api-key gsk_..."
        )

    run(
        api_key=args.api_key,
        input_path=args.input,
        output_path=args.output,
        low_conf_path=args.low_conf,
        checkpoint_path=args.checkpoint,
        sample=args.sample,
    )

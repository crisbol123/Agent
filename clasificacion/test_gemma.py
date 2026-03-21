import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
import time
import os
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv


ENV_FILE = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(dotenv_path=ENV_FILE)

HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError(f"No se encontro HF_TOKEN en {ENV_FILE}")

os.environ["HF_TOKEN"] = HF_TOKEN
os.environ["HF_HOME"] = r"D:\huggingface"
os.environ["HUGGINGFACE_HUB_CACHE"] = r"D:\huggingface\hub"
os.environ["TRANSFORMERS_CACHE"] = r"D:\huggingface\transformers"

HF_HOME_DIR = r"D:\huggingface"
HF_HUB_CACHE_DIR = r"D:\huggingface\hub"
HF_TRANSFORMERS_CACHE_DIR = r"D:\huggingface\transformers"


os.makedirs(HF_HUB_CACHE_DIR, exist_ok=True)
os.makedirs(HF_TRANSFORMERS_CACHE_DIR, exist_ok=True)

# Verificar que existe el dataset a evaluar
DATASET_FILE = os.path.join(os.path.dirname(__file__), "dataset_evaluacion_300_sin_general.csv")

if not os.path.exists(DATASET_FILE):
    raise FileNotFoundError(
        "\n" + "="*70 + "\n"
        "ERROR: Dataset no encontrado!\n\n"
        f"Archivo esperado: {DATASET_FILE}\n"
        + "="*70
    )

# Cargar dataset
print(f"Cargando dataset para evaluación: {DATASET_FILE}")
df = pd.read_csv(DATASET_FILE)
GROUND_TRUTH_COLUMN = 'decision_manual'
EXCLUDED_CATEGORY = 'GENERAL'

if GROUND_TRUTH_COLUMN not in df.columns:
    raise ValueError(
        f"La columna de ground truth '{GROUND_TRUTH_COLUMN}' no existe en el dataset. "
        f"Columnas disponibles: {list(df.columns)}"
    )

# Limpiar etiqueta del ground truth
df[GROUND_TRUTH_COLUMN] = df[GROUND_TRUTH_COLUMN].astype(str).str.strip().str.upper()

# Categorías objetivo (sin GENERAL)
CATEGORIES = ['ROUTING', 'SECURITY', 'QOS', 'CONNECTIVITY', 'MONITORING']

# Modelo a evaluar (solo Gemma)
MODEL_NAME = 'Gemma-2-7B-it'
MODEL_CONFIG = {
    'path': 'google/gemma-7b-it',
    'params': '7B'
}

# Prompt template para clasificación
CLASSIFICATION_PROMPT = """You are a network intent classifier for an IBN architecture.
Classify the question into exactly one category:

ROUTING, SECURITY, QOS, CONNECTIVITY, MONITORING

Taxonomy:
- ROUTING: routing protocols, route calculation, path selection.
- SECURITY: access control, authentication, encryption, AAA, PKI.
- QOS: traffic classification, marking, policing, shaping, queuing.
- CONNECTIVITY: interfaces, addressing, VLANs, NAT, tunnels, overlay setup, L2/L3 reachability.
- MONITORING: telemetry, measurement, logging, SNMP, performance tracking.

Rules:
- Do not classify based on keywords alone.
- Identify the underlying technical mechanism and its role in the network.
- Choose the SINGLE most relevant category based on primary intent.
- If multiple domains appear, pick the one the question is mainly about.


Your output must be exactly one token from this set: ROUTING, SECURITY, QOS, CONNECTIVITY, MONITORING.
Return only that label in uppercase, with no words before or after it.
Do not include explanations, punctuation, quotes, code fences, or line breaks.
Any output outside that exact label set is incorrect.

Question:
{question}
"""


def compute_robust_metrics(conf_matrix):
    """
    Métricas robustas para datasets desbalanceados.
    """
    cm = conf_matrix.astype(float)
    tp = np.diag(cm)

    support_true = cm.sum(axis=1)
    support_pred = cm.sum(axis=0)

    per_class_recall = np.divide(tp, support_true, out=np.zeros_like(tp), where=support_true > 0)
    per_class_precision = np.divide(tp, support_pred, out=np.zeros_like(tp), where=support_pred > 0)
    per_class_f1 = np.divide(
        2 * per_class_precision * per_class_recall,
        per_class_precision + per_class_recall,
        out=np.zeros_like(tp),
        where=(per_class_precision + per_class_recall) > 0,
    )

    macro_precision = float(np.mean(per_class_precision))
    macro_recall = float(np.mean(per_class_recall))
    macro_f1 = float(np.mean(per_class_f1))
    balanced_accuracy = macro_recall
    worst_class_recall = float(np.min(per_class_recall))
    robust_score = float(0.5 * macro_f1 + 0.3 * balanced_accuracy + 0.2 * worst_class_recall)

    return {
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'balanced_accuracy': balanced_accuracy,
        'worst_class_recall': worst_class_recall,
        'robust_score': robust_score,
        'per_class_precision': per_class_precision.tolist(),
        'per_class_recall': per_class_recall.tolist(),
        'per_class_f1': per_class_f1.tolist(),
    }

def load_model(model_name, model_config):
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch

        device = "cuda:0"
        
        if not torch.cuda.is_available():
            raise RuntimeError(
                "\nERROR: CUDA no disponible. Verifica que:\n"
                "  1. Tienes instalado PyTorch con soporte CUDA:\n"
                "     pip install torch --index-url https://download.pytorch.org/whl/cu124\n"
                "  2. Los drivers de NVIDIA están actualizados\n"
                "  3. Ejecuta: nvidia-smi para verificar la GPU"
            )
        
        print(f"\nCargando {model_name} en {torch.cuda.get_device_name(0)}...")
        print(f"  Cache HF: {HF_HUB_CACHE_DIR}")
        
        tokenizer = AutoTokenizer.from_pretrained(
            model_config['path'],
            trust_remote_code=model_config.get('trust_remote_code', False),
            cache_dir=HF_HUB_CACHE_DIR,
            token=os.environ["HF_TOKEN"],
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_config['path'],
            torch_dtype=torch.bfloat16,
            device_map=device,
            trust_remote_code=model_config.get('trust_remote_code', False),
            cache_dir=HF_HUB_CACHE_DIR,
            token=os.environ["HF_TOKEN"],
        )
        print(f"  VRAM usada: {torch.cuda.memory_allocated(0)/1024**3:.2f} GB")
        print(f"  Modelo {model_name} cargado exitosamente en GPU")
        return tokenizer, model
    except Exception as e:
        print(f"Error cargando {model_name}: {e}")
        return None, None

def classify_with_slm(question, tokenizer, model):
    try:
        messages = [
            {
                "role": "user",
                "content": CLASSIFICATION_PROMPT.format(question=question)
            }
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to("cuda:0")

        outputs = model.generate(
            **inputs,
            max_new_tokens=15,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

        raw = tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        ).strip().upper()

        # Buscar categoría al inicio
        for cat in CATEGORIES:
            if raw.startswith(cat):
                return cat

        # Buscar categoría en cualquier parte del texto
        for cat in CATEGORIES:
            if cat in raw:
                return cat

        print(f"[UNKNOWN OUTPUT] {raw}")
        return 'UNKNOWN'

    except Exception as e:
        print(f"Error en clasificación: {e}")
        return 'UNKNOWN'

def evaluate_model(model_name, model_config, df_sample):
    """
    Evalúa Gemma (prueba rápida)
    """
    print(f"\n{'='*80}")
    print(f"PRUEBA GEMMA: {model_name} ({model_config['params']})")
    print(f"{'='*80}")
    
    # Cargar modelo
    tokenizer, model = load_model(model_name, model_config)
    
    if tokenizer is None or model is None:
        print(f"Error: No se pudo cargar {model_name}")
        return
    
    # Clasificar cada ejemplo
    predictions = []
    start_time = time.time()
    
    for idx, row in df_sample.iterrows():
        if idx % 10 == 0:
            print(f"Procesando {idx}/{len(df_sample)}...")
        
        pred = classify_with_slm(row['question'], tokenizer, model)
        predictions.append(pred)
    
    elapsed_time = time.time() - start_time
    
    # Calcular métricas
    y_true = [str(cat).strip().upper() for cat in df_sample[GROUND_TRUTH_COLUMN].tolist()]
    y_pred = predictions
    
    unknown_count = predictions.count('UNKNOWN')
    if unknown_count > 0:
        print(f"  Advertencia: {unknown_count} muestras sin categoría detectada (UNKNOWN)")

    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0, labels=CATEGORIES
    )

    macro_precision_skl, macro_recall_skl, macro_f1_skl, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0, labels=CATEGORIES
    )

    # Reporte detallado
    report = classification_report(y_true, y_pred, labels=CATEGORIES, target_names=CATEGORIES, zero_division=0)
    conf_matrix = confusion_matrix(y_true, y_pred, labels=CATEGORIES)
    row_sums = conf_matrix.sum(axis=1, keepdims=True)
    conf_matrix_normalized = np.divide(
        conf_matrix.astype(float),
        row_sums,
        out=np.zeros_like(conf_matrix, dtype=float),
        where=row_sums > 0,
    )
    robust_metrics = compute_robust_metrics(conf_matrix)
    
    # Mostrar resultados en consola
    print(f"\n{'='*80}")
    print(f"RESULTADOS PARA {model_name}:")
    print(f"{'='*80}")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision (weighted): {precision:.4f}")
    print(f"  Recall (weighted): {recall:.4f}")
    print(f"  F1-Score (weighted): {f1:.4f}")
    print(f"  Macro-F1: {macro_f1_skl:.4f}")
    print(f"  Balanced Accuracy: {robust_metrics['balanced_accuracy']:.4f}")
    print(f"  Worst-class Recall: {robust_metrics['worst_class_recall']:.4f}")
    print(f"  Robust Score: {robust_metrics['robust_score']:.4f}")
    print(f"  Unknown rate: {unknown_count/len(df_sample):.2%}")
    print(f"  Tiempo total: {elapsed_time:.2f}s")
    print(f"  Tiempo promedio por muestra: {elapsed_time/len(df_sample):.2f}s")
    
    print(f"\n{'='*80}")
    print("CLASIFICACIÓN POR CATEGORÍA:")
    print(f"{'='*80}")
    print(report)
    
    print(f"\n{'='*80}")
    print("MATRIZ DE CONFUSIÓN (Normalizada):")
    print(f"{'='*80}")
    print("Predicho →")
    print("Real ↓")
    # Encabezados
    print("  " + "  ".join(f"{cat:12}" for cat in CATEGORIES))
    for i, cat in enumerate(CATEGORIES):
        print(f"{cat:2}", end=" ")
        for j in range(len(CATEGORIES)):
            print(f"{conf_matrix_normalized[i][j]:12.2%}", end="")
        print()

def main():
    """
    Función principal - Prueba rápida de Gemma
    """
    print("="*80)
    print("PRUEBA RÁPIDA - EVALUACION GEMMA")
    print("="*80)
    print(f"Dataset: {DATASET_FILE}")
    
    # Mostrar distribución real del dataset
    print(f"\nDataset cargado: {len(df)} muestras")
    print("Distribución por categoría:")
    category_distribution = df[GROUND_TRUTH_COLUMN].value_counts().sort_index()
    print(category_distribution)
    print()
    
    # Evaluar Gemma
    evaluate_model(MODEL_NAME, MODEL_CONFIG, df)
    
    print(f"\n{'='*80}")
    print("PRUEBA COMPLETADA - Sin exportación de JSON")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()

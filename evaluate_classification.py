import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
import json
import time
import os
from datetime import datetime

# Verificar que existe el dataset balanceado
BALANCED_DATASET_FILE = "dataset_balanced_evaluation.csv"

if not os.path.exists(BALANCED_DATASET_FILE):
    raise FileNotFoundError(
        "\n" + "="*70 + "\n"
        "ERROR: Dataset balanceado no encontrado!\n\n"
        "Por favor ejecuta primero el siguiente comando:\n"
        "    python create_balanced_dataset.py\n\n"
        "Esto generará 'dataset_balanced_evaluation.csv' con 800 muestras\n"
        "(100 por categoría, 8 categorías técnicas) necesarias para una comparación justa.\n"
        + "="*70
    )

# Cargar dataset balanceado
print("Cargando dataset balanceado para evaluación...")
df = pd.read_csv(BALANCED_DATASET_FILE)

# Definir las categorías (sin OTHERS - no aporta valor técnico)
CATEGORIES = ['DIAG', 'QoS', 'RP', 'TN', 'INF', 'MNG', 'PKI', 'ACL']

# Configuración de modelos a evaluar
MODELS = {
    'Qwen2-7B-Instruct': {
        'path': 'Qwen/Qwen2-7B-Instruct',
        'params': '7B'
    },
    'Llama-3.1-8B-Instruct': {
        'path': 'meta-llama/Meta-Llama-3.1-8B-Instruct',
        'params': '8B'
    },
    'Gemma-7B-Instruct': {
        'path': 'google/gemma-7b-it',
        'params': '7B'
    },
    'Yi-6B-Chat': {
        'path': '01-ai/Yi-6B-Chat',
        'params': '6B'
    },
    'DeciLM-7B-Instruct': {
        'path': 'Deci/DeciLM-7B-instruct',
        'params': '7B'
    }
}

# Prompt template para clasificación
CLASSIFICATION_PROMPT = """You are a network configuration expert. Classify the following Cisco IOS configuration into ONE of these categories:

Categories:
- DIAG: Diagnostic and operational commands (show, display, enable)
- QoS: Quality of Service configurations
- RP: Routing Protocols (OSPF, BGP, EIGRP, etc.)
- TN: Tunnels and VPN (IPsec, GRE, etc.)
- INF: Infrastructure (interfaces, VLANs, NAT)
- MNG: Management and Monitoring (SNMP, NetFlow, IP SLA)
- PKI: Security and PKI (certificates, AAA, RADIUS)
- ACL: Access Control Lists and Security

Question: {question}
Answer: {answer}

Respond with ONLY the category name (e.g., "DIAG" or "QoS"). No explanation needed."""

def load_model(model_name, model_config):
    """
    Carga un modelo SLM usando transformers
    """
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
        
        print(f"\nCargando {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_config['path'])
        model = AutoModelForCausalLM.from_pretrained(
            model_config['path'],
            torch_dtype=torch.float16,
            device_map="auto"
        )
        print(f"Modelo {model_name} cargado exitosamente")
        return tokenizer, model
    except Exception as e:
        print(f"Error cargando {model_name}: {e}")
        return None, None

def classify_with_slm(question, answer, tokenizer, model):
    """
    Clasifica una pregunta-respuesta usando un SLM
    """
    try:
        prompt = CLASSIFICATION_PROMPT.format(question=question, answer=answer)  # Sin límite
        
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=10,
            temperature=0.1,
            do_sample=False
        )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extraer solo la categoría de la respuesta
        response = response.split(prompt)[-1].strip().upper()
        
        # Validar que sea una categoría válida
        for cat in CATEGORIES:
            if cat in response:
                return cat
        
        return 'DIAG'  # Default si no se reconoce
        
    except Exception as e:
        print(f"Error en clasificación: {e}")
        return 'DIAG'

def evaluate_model(model_name, model_config, df_sample):
    """
    Evalúa un modelo completo
    """
    print(f"\n{'='*80}")
    print(f"EVALUANDO: {model_name} ({model_config['params']})")
    print(f"{'='*80}")
    
    # Cargar modelo
    tokenizer, model = load_model(model_name, model_config)
    
    if tokenizer is None or model is None:
        print(f"Saltando {model_name} - No se pudo cargar")
        return None
    
    # Clasificar cada ejemplo
    predictions = []
    start_time = time.time()
    
    for idx, row in df_sample.iterrows():
        if idx % 10 == 0:
            print(f"Procesando {idx}/{len(df_sample)}...")
        
        pred = classify_with_slm(row['question'], row['answer'], tokenizer, model)
        predictions.append(pred)
    
    elapsed_time = time.time() - start_time
    
    # Calcular métricas
    y_true = df_sample['ground_truth_category'].tolist()
    y_pred = predictions
    
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )
    
    # Reporte detallado
    report = classification_report(y_true, y_pred, target_names=CATEGORIES, zero_division=0)
    conf_matrix = confusion_matrix(y_true, y_pred, labels=CATEGORIES)
    
    results = {
        'model_name': model_name,
        'params': model_config['params'],
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'elapsed_time': elapsed_time,
        'samples_evaluated': len(df_sample),
        'avg_time_per_sample': elapsed_time / len(df_sample),
        'classification_report': report,
        'confusion_matrix': conf_matrix.tolist()
    }
    
    print(f"\nResultados para {model_name}:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    print(f"  Tiempo total: {elapsed_time:.2f}s")
    print(f"  Tiempo promedio por muestra: {elapsed_time/len(df_sample):.2f}s")
    
    return results

def main():
    """
    Función principal de evaluación
    """
    print("="*80)
    print("EVALUACION DE CLASIFICACION - 5 SLMs")
    print("="*80)
    
    # El dataset ya está balanceado (100 muestras por categoría)
    print(f"\nDataset cargado: {len(df)} muestras")
    print(f"Distribución por categoría:")
    print(df['ground_truth_category'].value_counts().sort_index())
    print()
    
    # Evaluar cada modelo
    all_results = []
    
    for model_name, model_config in MODELS.items():
        result = evaluate_model(model_name, model_config, df)
        if result:
            all_results.append(result)
    
    # Guardar resultados
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"classification_results_{timestamp}.json"
    
    # Agregar información del dataset usado
    evaluation_info = {
        'timestamp': timestamp,
        'total_samples': len(df),
        'samples_per_category': 100,
        'categories': CATEGORIES,
        'models_evaluated': list(MODELS.keys()),
        'results': all_results
    }
    
    with open(results_file, 'w') as f:
        json.dump(evaluation_info, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"RESULTADOS GUARDADOS EN: {results_file}")
    print(f"{'='*80}")
    
    # Mostrar comparación final
    print("\nCOMPARACION FINAL:")
    print(f"{'Modelo':<30} {'Accuracy':<12} {'F1-Score':<12} {'Tiempo (s)':<12}")
    print("-"*80)
    
    for result in sorted(all_results, key=lambda x: x['f1_score'], reverse=True):
        print(f"{result['model_name']:<30} {result['accuracy']:<12.4f} {result['f1_score']:<12.4f} {result['elapsed_time']:<12.2f}")

if __name__ == "__main__":
    main()

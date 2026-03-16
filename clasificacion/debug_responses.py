import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM

CATEGORIES = ['DIAG', 'QOS', 'RP', 'TN', 'INF', 'MNG', 'PKI', 'ACL']

CLASSIFICATION_PROMPT = """You are a Cisco networking expert.

Classify the following Cisco IOS question into ONE category.

Return ONLY the category label from this list:

DIAG
QOS
RP
TN
INF
MNG
PKI
ACL

Rules:
- Output ONLY the label
- Do NOT explain
- Do NOT write sentences
- Do NOT output anything else

Question:
{question}

Label:"""

df = pd.read_csv("dataset_balanced_evaluation.csv")
sample = df.head(10)

MODEL_PATH = "Qwen/Qwen2-7B-Instruct"
print(f"Cargando {MODEL_PATH}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype=torch.float16, device_map="cuda:0", trust_remote_code=True)

print("\n" + "="*80)
print("RESPUESTAS CRUDAS DEL MODELO")
print("="*80)

for idx, row in sample.iterrows():
    # Usar chat template correcto para modelos instruct
    messages = [
        {
            "role": "user",
            "content": CLASSIFICATION_PROMPT.format(question=row['question'])
        }
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to("cuda:0")
    
    print(f"\nMuestra {idx}:")
    print(f"  Ground truth:   {row['ground_truth_category']}")
    print(f"  Tokens entrada: {inputs['input_ids'].shape[1]}")
    print(f"  Prompt enviado:\n{prompt}\n{'-'*40}")
   
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=15,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )
    
    raw = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip().upper()
    found = next((cat for cat in CATEGORIES if raw.startswith(cat)), None)
    if not found:
        found = next((cat for cat in CATEGORIES if cat in raw), 'UNKNOWN')
    
    print(f"  Respuesta cruda: '{raw}'")
    print(f"  Categoría extraída: {found}")
    print(f"  Ground truth upper: {row['ground_truth_category'].upper()}")

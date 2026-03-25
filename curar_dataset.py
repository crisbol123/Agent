import pandas as pd
import re

# --- CARGAR ---
df = pd.read_csv('requirements_classified_gpt5.csv')

# --- FILTRAR: solo con código, descartar GENERAL ---
df = df[df['answer'].str.contains('```', na=False)].copy()
df = df[df['category'] != 'GENERAL'].copy()

# --- LIMPIAR prefijos de preguntas ---
def clean_question(q):
    q = re.sub(r'^\s*\*+\s*', '', str(q))                          # quita ** al inicio
    q = re.sub(r'^\s*Question:\*+\s*', '', q, flags=re.IGNORECASE) # quita Question:**
    q = re.sub(r'^\s*Question:\s*', '', q, flags=re.IGNORECASE)    # quita Question:
    return q.strip()

df['question'] = df['question'].apply(clean_question)

# --- MUESTREO ESTRATIFICADO: 20 por categoría ---
categories = ['SECURITY', 'ROUTING', 'MONITORING', 'QOS', 'CONNECTIVITY']
df_filtered = df[df['category'].isin(categories)].copy()

frames = []
for cat in categories:
    subset = df_filtered[df_filtered['category'] == cat]
    frames.append(subset.sample(n=20, random_state=42))

sample = pd.concat(frames).reset_index(drop=True)

# --- GUARDAR ---
sample[['id', 'question', 'answer', 'category']].to_csv('dataset_100_curado.csv', index=False)

print('Total muestras:', len(sample))
print(sample['category'].value_counts())

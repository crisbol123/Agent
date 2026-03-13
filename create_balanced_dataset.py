import pandas as pd

# Cargar dataset con ground truth
print("Cargando dataset con ground truth...")
df = pd.read_csv("dataset_classified_ground_truth.csv")

# Definir las categorías (sin OTHERS - no aporta valor técnico)
CATEGORIES = ['DIAG', 'QoS', 'RP', 'TN', 'INF', 'MNG', 'PKI', 'ACL']

def create_balanced_sample(df, samples_per_category=100):
    """
    Crea una muestra balanceada tomando N muestras de cada categoría
    """
    balanced_samples = []
    
    print(f"\nCreando muestra balanceada con {samples_per_category} muestras por categoria...")
    print("=" * 80)
    
    for category in CATEGORIES:
        category_df = df[df['ground_truth_category'] == category]
        n_available = len(category_df)
        n_to_sample = min(samples_per_category, n_available)
        
        if n_available < samples_per_category:
            print(f"  {category:10s}: {n_available:4d} disponibles (usando todas)")
            sampled = category_df
        else:
            print(f"  {category:10s}: {n_to_sample:4d} de {n_available:4d} disponibles")
            sampled = category_df.sample(n=n_to_sample, random_state=42)
        
        balanced_samples.append(sampled)
    
    balanced_df = pd.concat(balanced_samples, ignore_index=True)
    
    # Shuffle para mezclar categorías
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print("=" * 80)
    print(f"Total de muestras en dataset balanceado: {len(balanced_df)}")
    
    return balanced_df

# Crear muestra balanceada
SAMPLES_PER_CATEGORY = 100
balanced_df = create_balanced_sample(df, SAMPLES_PER_CATEGORY)

# Guardar dataset balanceado
output_file = "dataset_balanced_evaluation.csv"
balanced_df.to_csv(output_file, index=False)

print(f"\nDataset balanceado guardado en: {output_file}")

# Verificar distribución final
print("\nDistribucion final del dataset balanceado:")
print("=" * 80)
distribution = balanced_df['ground_truth_category'].value_counts().sort_index()
for category, count in distribution.items():
    percentage = (count / len(balanced_df)) * 100
    print(f"  {category:10s}: {count:4d} ({percentage:5.1f}%)")

print("=" * 80)
print(f"Total: {len(balanced_df)} muestras")

import pandas as pd
import os

os.chdir("clasificacion")

# Cargar ambos archivos
df_ollama = pd.read_csv("requirements_classified_v3.csv")
df_openai = pd.read_csv("requirements_classified_gpt5.csv")

print("=" * 80)
print("COMPARACIÓN DE CLASIFICACIONES: Ollama (v3) vs OpenAI (GPT-5)")
print("=" * 80)

# Verificar que sean los mismos datos
assert len(df_ollama) == len(df_openai), "Archivos con diferente número de filas"
assert (df_ollama["question"] == df_openai["question"]).all(), "Las preguntas no coinciden"

print(f"\n✓ Ambos archivos tienen {len(df_ollama)} preguntas en el mismo orden")

# Crear columna de comparación
df_compare = df_ollama[["id", "question"]].copy()
df_compare["ollama"] = df_ollama["category"]
df_compare["openai"] = df_openai["category"]
df_compare["match"] = df_compare["ollama"] == df_compare["openai"]

# Estadísticas
total = len(df_compare)
matches = df_compare["match"].sum()
mismatches = total - matches
accuracy = (matches / total) * 100

print(f"\n📊 RESULTADOS:")
print(f"   Total preguntas    : {total}")
print(f"   Coincidencias ✓    : {matches} ({accuracy:.1f}%)")
print(f"   Desacuerdos ✗      : {mismatches} ({100-accuracy:.1f}%)")

# Distribución de coincidencias por categoría
print(f"\n📋 COINCIDENCIAS POR CATEGORÍA:")
cats = df_compare[df_compare["match"]]["ollama"].value_counts().sort_index()
for cat, count in cats.items():
    pct = (count / matches) * 100
    print(f"   {cat:15s}: {count:5d} ({pct:5.1f}%)")

# Desacuerdos por categoría
print(f"\n⚠️  DESACUERDOS POR CATEGORÍA (Ollama):")
disagree = df_compare[~df_compare["match"]]
cats_disagree = disagree["ollama"].value_counts().sort_index()
for cat, count in cats_disagree.items():
    pct = (count / mismatches) * 100
    print(f"   {cat:15s}: {count:5d} ({pct:5.1f}%)")

# Guardar resultados
print(f"\n💾 Guardando resultados...")

# Preguntas donde COINCIDEN
df_match = df_compare[df_compare["match"]][["id", "question", "ollama"]].copy()
df_match.columns = ["id", "question", "category"]
df_match.to_csv("comparacion_coincidencias.csv", index=False)
print(f"   ✓ comparacion_coincidencias.csv ({len(df_match)} filas)")

# Preguntas donde DIFIEREN
df_disagree = df_compare[~df_compare["match"]][["id", "question", "ollama", "openai"]].copy()
df_disagree.columns = ["id", "question", "ollama", "openai"]
df_disagree.to_csv("comparacion_desacuerdos.csv", index=False)
print(f"   ✓ comparacion_desacuerdos.csv ({len(df_disagree)} filas)")

# Resumen completo
df_compare.to_csv("comparacion_completa.csv", index=False)
print(f"   ✓ comparacion_completa.csv ({len(df_compare)} filas)")

print(f"\n✅ Comparación completada!")
print(f"\nPrimeras coincidencias:")
print(df_match.head(3).to_string(index=False))
print(f"\nPrimeros desacuerdos:")
print(df_disagree.head(3).to_string(index=False))

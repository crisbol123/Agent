from pathlib import Path

import matplotlib.pyplot as plt

# Datos fijos proporcionados por el usuario (en GB).
base_dir = Path(__file__).resolve().parent
model_sizes_gb = {
    "Gemma-2-9B-it": 17.2,
    "Llama-3.1-8B-Instruct": 17.0,
    "Qwen2.5-7B-Instruct": 14.1,
    "Zephyr-7B": 13.4,
    "FLAN-T5-large": 2.91,
    "FLAN-T5-base": 0.947,  # 947 MB
}

# Ordena de mayor a menor para mostrar de izquierda a derecha los que mas ocupan.
items_ordenados = sorted(model_sizes_gb.items(), key=lambda x: x[1], reverse=True)
labels = [name for name, _ in items_ordenados]
uso_disco_gb = [size for _, size in items_ordenados]

# Configuracion de colores
palette = ["#0F766E", "#14B8A6", "#2DD4BF", "#0D9488", "#5EEAD4", "#115E59"]
colores = [palette[i % len(palette)] for i in range(len(uso_disco_gb))]

# Crear el grafico de barras
fig, ax = plt.subplots(figsize=(13, 7))
barras = ax.bar(labels, uso_disco_gb, color=colores)

# Anadir etiquetas y titulo
ax.set_ylabel("Espacio en disco (GB)", fontsize=12)
ax.set_title("Comparacion de Espacio en Disco por Modelo (SLMs)", fontsize=14, fontweight="bold")
ax.set_ylim(0, max(uso_disco_gb) + 2)

# Mostrar el valor numerico en cada barra
for barra in barras:
    yval = barra.get_height()
    valor_txt = f"{yval:.2f} GB" if yval >= 1 else f"{yval * 1024:.0f} MB"
    ax.text(
        barra.get_x() + barra.get_width() / 2,
        yval + 0.2,
        valor_txt,
        ha="center",
        va="bottom",
        fontweight="bold",
    )

# Mejorar el diseno y la visibilidad
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.xticks(rotation=12, ha="right")
plt.tight_layout()

# Guardar siempre el grafico a archivo (util si el backend no abre ventana)
output_path = base_dir / "grafica_espacio_disco_slms.png"
plt.savefig(output_path, dpi=200, bbox_inches="tight")
print(f"Grafico guardado en: {output_path}")
print("Fuente de datos: valores manuales definidos en el script")

# Intentar mostrar el grafico en pantalla si hay backend interactivo
backend = plt.get_backend().lower()
interactive_backends = ("macosx", "qt", "tk", "gtk", "wx")

if any(name in backend for name in interactive_backends):
    plt.show()
else:
    print(f"Backend actual ('{backend}') no es interactivo; abre el PNG generado.")

plt.close(fig)

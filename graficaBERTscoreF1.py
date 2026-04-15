import matplotlib.pyplot as plt
import numpy as np

# Datos (REEMPLAZA con tus valores reales)
quantizations = ['INT4', 'INT8', 'Float16']

qwen_scores = [0.985, 0.988, 0.989]
llama_scores = [0.944, 0.9852,0.985]

x = np.arange(len(quantizations))
width = 0.35
all_scores = qwen_scores + llama_scores
y_min = min(all_scores) - 0.02
y_max = max(all_scores) + 0.01

plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(9, 5), dpi=120)

bars1 = ax.bar(
    x - width/2,
    qwen_scores,
    width,
    label='Qwen2.5-7B-Instruct',
    color='#2E86AB',
    edgecolor='white',
    linewidth=1.2
)
bars2 = ax.bar(
    x + width/2,
    llama_scores,
    width,
    label='LLaMA 3.1 8B Instruct',
    color='#F18F01',
    edgecolor='white',
    linewidth=1.2
)

# Función para poner solo el valor arriba de cada barra
def add_labels(bars):
    y_offset = (y_max - y_min) * 0.015
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2,
            height + y_offset,
            f'{height:.2f}',
            ha='center',
            va='bottom',
            fontsize=9,
            fontweight='semibold'
        )

add_labels(bars1)
add_labels(bars2)

ax.set_xlabel('Cuantización', fontsize=11)
ax.set_ylabel('BERTScore F1', fontsize=11)
ax.set_title('Comparación de BERTScore F1 por Cuantización', fontsize=13, fontweight='bold', pad=14)
ax.set_xticks(x, quantizations)
ax.set_ylim(y_min, y_max)

ax.legend(loc='upper left', frameon=True)
ax.grid(axis='y', linestyle='--', alpha=0.35)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

fig.tight_layout()

plt.show()
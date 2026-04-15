import matplotlib.pyplot as plt
import numpy as np

# Datos (REEMPLAZA con tus valores reales de ROUGE-L)
quantizations = ['INT4', 'INT8', 'Float16']

qwen_rouge_l = [0.874, 0.888, 0.890]
llama_rouge_l = [0.8307, 0.8613, 0.857]

x = np.arange(len(quantizations))
width = 0.35
all_scores = qwen_rouge_l + llama_rouge_l
y_min = max(0, min(all_scores) - 0.03)
y_max = min(1, max(all_scores) + 0.02)

plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(9, 5), dpi=120)

bars1 = ax.bar(
    x - width/2,
    qwen_rouge_l,
    width,
    label='Qwen2.5-7B-Instruct',
    color='#2E86AB',
    edgecolor='white',
    linewidth=1.2
)
bars2 = ax.bar(
    x + width/2,
    llama_rouge_l,
    width,
    label='LLaMA 3.1 8B Instruct',
    color='#F18F01',
    edgecolor='white',
    linewidth=1.2
)


def add_labels(bars):
    y_offset = (y_max - y_min) * 0.015
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
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
ax.set_ylabel('ROUGE-L', fontsize=11)
ax.set_title('Comparación de ROUGE-L por Cuantización', fontsize=13, fontweight='bold', pad=14)
ax.set_xticks(x, quantizations)
ax.set_ylim(y_min, y_max)

ax.legend(loc='upper left', frameon=True)
ax.grid(axis='y', linestyle='--', alpha=0.35)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

fig.tight_layout()
plt.show()
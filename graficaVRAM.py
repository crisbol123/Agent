from pathlib import Path

import matplotlib.pyplot as plt

# Datos hipotéticos de uso de VRAM (en GB) para un modelo de 8B de parámetros
# Estos valores son representativos de la reducción de memoria
# que se logra con las diferentes técnicas de cuantización.
metodos_cuantizacion = ['4-bit', '8-bit', 'FP16']
uso_vram_gb = [4.17, 8.11, 11.9]  # Valores en Gigabytes (GB)

# Configuración de colores
colores = ['#1E3A8A', '#2563EB', '#60A5FA'] # Azules para representar el modelo

# Crear el gráfico de barras
fig, ax = plt.subplots(figsize=(10, 6))
barras = ax.bar(metodos_cuantizacion, uso_vram_gb, color=colores)

# Añadir etiquetas y título
ax.set_ylabel('Uso de VRAM (GB)', fontsize=12)
ax.set_title('Comparación de Uso de VRAM por Método de Cuantización', fontsize=14, fontweight='bold')
ax.set_ylim(0, max(uso_vram_gb) + 2) # Ajustar límite Y

# Mostrar el valor numérico en cada barra
for barra in barras:
    yval = barra.get_height()
    ax.text(barra.get_x() + barra.get_width()/2, yval + 0.3, f'{yval:.1f} GB', ha='center', va='bottom', fontweight='bold')

# Mejorar el diseño y la visibilidad
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xticks(rotation=0)
plt.tight_layout()

# Guardar siempre el gráfico a archivo (útil si el backend no abre ventana)
output_path = Path(__file__).resolve().parent / 'grafica_vram.png'
plt.savefig(output_path, dpi=200, bbox_inches='tight')
print(f'Gráfico guardado en: {output_path}')

# Intentar mostrar el gráfico en pantalla si hay backend interactivo
backend = plt.get_backend().lower()
interactive_backends = ('macosx', 'qt', 'tk', 'gtk', 'wx')

if any(name in backend for name in interactive_backends):
    plt.show()
else:
    print(f"Backend actual ('{backend}') no es interactivo; abre el PNG generado.")

plt.close(fig)
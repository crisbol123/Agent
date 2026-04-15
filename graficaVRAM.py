from pathlib import Path
import json

import matplotlib.pyplot as plt

# Busca automaticamente el reporte mas reciente generado por measure_model_resources.py
base_dir = Path(__file__).resolve().parent
reportes = sorted(base_dir.glob("model_resource_report_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)

if not reportes:
    raise FileNotFoundError(
        "No se encontraron archivos model_resource_report_*.json en el directorio del proyecto. "
        "Ejecuta primero: python measure_model_resources.py"
    )

reporte_path = reportes[0]
with open(reporte_path, "r", encoding="utf-8") as f:
    data = json.load(f)

resultados_ok = [
    r for r in data.get("results", [])
    if r.get("status") == "ok" and r.get("vram_peak_during_load_gb") is not None
]

if not resultados_ok:
    raise ValueError("El reporte no tiene modelos con VRAM valida para graficar.")

# Ordena de mayor a menor VRAM para mostrar de izquierda a derecha los que mas consumen.
resultados_ok = sorted(
    resultados_ok,
    key=lambda r: float(r["vram_peak_during_load_gb"]),
    reverse=True,
)

modelos = [r["model_name"] for r in resultados_ok]
uso_vram_gb = [float(r["vram_peak_during_load_gb"]) for r in resultados_ok]

# Etiquetas cortas para mejorar legibilidad en eje X.
labels = []
for r in resultados_ok:
    name = r["model_name"]
    params = r.get("params", "")
    labels.append(f"{name}\n({params})" if params else name)

# Configuración de colores
palette = ["#1E3A8A", "#2563EB", "#60A5FA", "#3B82F6", "#1D4ED8", "#93C5FD"]
colores = [palette[i % len(palette)] for i in range(len(uso_vram_gb))]

# Crear el gráfico de barras
fig, ax = plt.subplots(figsize=(13, 7))
barras = ax.bar(labels, uso_vram_gb, color=colores)

# Añadir etiquetas y título
ax.set_ylabel('Uso de VRAM (GB)', fontsize=12)
ax.set_title('Comparación de VRAM Pico por Modelo (SLMs)', fontsize=14, fontweight='bold')
ax.set_ylim(0, max(uso_vram_gb) + 2) # Ajustar límite Y

# Mostrar el valor numérico en cada barra
for barra in barras:
    yval = barra.get_height()
    ax.text(barra.get_x() + barra.get_width()/2, yval + 0.2, f'{yval:.2f} GB', ha='center', va='bottom', fontweight='bold')

# Mejorar el diseño y la visibilidad
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xticks(rotation=12, ha='right')
plt.tight_layout()

# Guardar siempre el gráfico a archivo (útil si el backend no abre ventana)
output_path = base_dir / 'grafica_vram_slms.png'
plt.savefig(output_path, dpi=200, bbox_inches='tight')
print(f'Gráfico guardado en: {output_path}')
print(f'Reporte usado: {reporte_path.name}')

# Intentar mostrar el gráfico en pantalla si hay backend interactivo
backend = plt.get_backend().lower()
interactive_backends = ('macosx', 'qt', 'tk', 'gtk', 'wx')

if any(name in backend for name in interactive_backends):
    plt.show()
else:
    print(f"Backend actual ('{backend}') no es interactivo; abre el PNG generado.")

plt.close(fig)
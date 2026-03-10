import pandas as pd
import sys

# 1. Cargamos el dataset
print("Cargando dataset...", flush=True)
df = pd.read_csv("requirements_questions_v2.csv")
print(f"Dataset cargado: {len(df)} filas\n", flush=True)

# 2. Usamos la MISMA función de clasificación que Clasifier.py
def classify_intent(row):
    text = str(row['answer']).lower()
    
    # Prioridad 1: DIAG (Comandos de visualización/operativos)
    if "show " in text or "display " in text:
        return 'DIAG (Diagnostics)'
    
    # Prioridad 2: TN (Tunnels & VPN)
    if any(kw in text for kw in ['tunnel', 'gre', 'ipsec', 'vpn', 'isakmp', 'crypto map']):
        return 'TN (Tunnels)'
    
    # Prioridad 3: QoS (Quality of Service)
    if any(kw in text for kw in ['policy-map', 'class-map', 'police', 'priority', 'shape ', 'bandwidth', 'rtp', 'header-compression']):
        return 'QoS (Quality of Service)'
    
    # Prioridad 4: ACL (Security)
    if any(kw in text for kw in ['access-list', 'permit', 'deny', 'access-group', 'firewall', 'zone-policy', 'zone ']):
        return 'ACL (Security)'
    
    # Prioridad 5: RP (Routing)
    if any(kw in text for kw in ['ospf', 'bgp', 'rip', 'eigrp', 'router ', 'ip route', 'next-hop', 'hsrp', 'igmp', 'multicast']):
        return 'RP (Routing)'
    
    # Prioridad 6: MNG (Management & Performance)
    if any(kw in text for kw in ['ip sla', 'netflow', 'snmp', 'logging', 'ntp', 'http server', 'monitor', 'lldp', 'ccm', 'continuity-check']):
        return 'MNG (Management)'
    
    # Prioridad 7: PKI (Security & PKI)
    if any(kw in text for kw in ['crypto ca', 'certificate', 'cert-', 'radius', 'aaa ', 'pki']):
        return 'PKI (Security & PKI)'
    
    # Prioridad 8: INF (Infrastructure & L2)
    if any(kw in text for kw in ['interface ', 'ip address', 'encapsulation', 'frame-relay', 'dot1q', 'arp', 'nat64']):
        return 'INF (Infrastructure)'
    
    return 'OTHERS'

print("Clasificando elementos...", flush=True)
df['category'] = df.apply(classify_intent, axis=1)

# 3. Filtramos solo los OTHERS
others_df = df[df['category'] == 'OTHERS']

print(f"Total de elementos en OTHERS: {len(others_df)}", flush=True)
print("-" * 50, flush=True)

# 4. Mostramos una muestra aleatoria de 50 para inspección manual
muestra = others_df.sample(n=min(len(others_df), 50), random_state=42)

for i, (idx, row) in enumerate(muestra.iterrows(), 1):
    print(f"MUESTRA #{i} (Fila original: {idx})", flush=True)
    print(f"PREGUNTA: {row['question']}", flush=True)
    print(f"RESPUESTA COMPLETA:\n{row['answer']}", flush=True)
    print("=" * 80, flush=True)

others_df.to_csv("solo_others_para_revisar.csv", index=False)
print("\n✅ Archivo guardado: solo_others_para_revisar.csv", flush=True)
print(f"✅ Total de OTHERS exportados: {len(others_df)}", flush=True)

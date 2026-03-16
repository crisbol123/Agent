import pandas as pd
import matplotlib.pyplot as plt

# 1. Cargar el dataset
file_path = "requirements_questions_v2.csv"
df = pd.read_csv(file_path)

def classify_intent(row):
    text = str(row['answer']).lower()
    
    # Prioridad 1: DIAG (Comandos de visualización/operativos)
    if any(kw in text for kw in ['show ', 'display ', 'enable', 'device> enable', 'device# end', 'privileged exec']):
        return 'DIAG (Diagnostics)'
    
    # Prioridad 2: TN (Tunnels & VPN)
    if any(kw in text for kw in ['tunnel', 'gre', 'ipsec', 'vpn', 'isakmp', 'crypto map']):
        return 'TN (Tunnels)'
    
    # Prioridad 3: QoS (Quality of Service)
    if any(kw in text for kw in ['policy-map', 'class-map', 'police', 'priority', 'shape ', 'bandwidth', 'rtp', 'header-compression', 'queue-limit', 'voice-class codec']):
        return 'QoS (Quality of Service)'
    
    # Prioridad 4: ACL (Security)
    if any(kw in text for kw in ['access-list', 'permit', 'deny', 'access-group', 'firewall', 'zone-policy', 'zone ', 'tcp synwait', 'alg vtcp', 'sip alg', 'sccp alg', 'ip nat service']):
        return 'ACL (Security)'
    
    # Prioridad 5: RP (Routing)
    if any(kw in text for kw in ['ospf', 'bgp', 'rip', 'eigrp', 'router ', 'ip route', 'next-hop', 'hsrp', 'igmp', 'multicast', 'mpls', 'nhrp', 'msdp', 'pim ', 'bidir-upstream']):
        return 'RP (Routing)'
    
    # Prioridad 6: MNG (Management & Performance)
    if any(kw in text for kw in ['ip sla', 'netflow', 'snmp', 'logging', 'ntp', 'http server', 'monitor', 'lldp', 'ccm', 'continuity-check', 'service-group', 'nbar', 'protocol-pack', 'ethernet cfm', 'beacon interval', 'bfd interval', 'bfd neighbor', 'traceroute-cache', 'statistics-distribution', 'packet capture', 'schedule periodic']):
        return 'MNG (Management)'
    
    # Prioridad 7: PKI (Security & PKI)
    if any(kw in text for kw in ['crypto ca', 'certificate', 'cert-', 'radius', 'aaa ', 'pki', 'method-est', 'authentication md5']):
        return 'PKI (Security & PKI)'
    
    # Prioridad 8: INF (Infrastructure & L2)
    if any(kw in text for kw in ['interface ', 'ip address', 'encapsulation', 'frame-relay', 'dot1q', 'arp', 'nat64', 'bridge-domain', 'map-t', 'nptv6', 'ipv6 unicast-routing', 'nat66', 'ip vrf', 'ip nat inside', 'ip nat outside']):
        return 'INF (Infrastructure)'
    
    return 'OTHERS'

# Aplicar la clasificación
df['category'] = df.apply(classify_intent, axis=1)

# 2. Generar Estadísticas
stats = df['category'].value_counts()
percentages = df['category'].value_counts(normalize=True) * 100

# 3. Mostrar Resultados para la Monografía
print("--- ANÁLISIS DE CATEGORÍAS PARA MONOGRAFÍA ---")
summary = pd.DataFrame({'Cantidad': stats, 'Porcentaje (%)': percentages})
print(summary)

# 4. Generar Gráfico para el documento
plt.figure(figsize=(10, 6))
stats.plot(kind='bar', color='skyblue', edgecolor='black')
plt.title('Distribución de Requerimientos Técnicos en el Dataset SLM_netconfig')
plt.xlabel('Categoría de Intención')
plt.ylabel('Número de Elementos')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('distribucion_dataset.png')
print("\n[Gráfico guardado como 'distribucion_dataset.png']")
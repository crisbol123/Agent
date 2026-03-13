import pandas as pd

# Cargar el dataset
file_path = "requirements_questions_v2.csv"
df = pd.read_csv(file_path)

def classify_intent(row):
    text = str(row['answer']).lower()
    
    if any(kw in text for kw in ['show ', 'display ', 'enable', 'device> enable', 'device# end', 'privileged exec']):
        return 'DIAG'
    
    if any(kw in text for kw in ['tunnel', 'gre', 'ipsec', 'vpn', 'isakmp', 'crypto map']):
        return 'TN'
    
    if any(kw in text for kw in ['policy-map', 'class-map', 'police', 'priority', 'shape ', 'bandwidth', 'rtp', 'header-compression', 'queue-limit', 'voice-class codec']):
        return 'QoS'
    
    if any(kw in text for kw in ['access-list', 'permit', 'deny', 'access-group', 'firewall', 'zone-policy', 'zone ', 'tcp synwait', 'alg vtcp', 'sip alg', 'sccp alg', 'ip nat service']):
        return 'ACL'
    
    if any(kw in text for kw in ['ospf', 'bgp', 'rip', 'eigrp', 'router ', 'ip route', 'next-hop', 'hsrp', 'igmp', 'multicast', 'mpls', 'nhrp', 'msdp', 'pim ', 'bidir-upstream']):
        return 'RP'
    
    if any(kw in text for kw in ['ip sla', 'netflow', 'snmp', 'logging', 'ntp', 'http server', 'monitor', 'lldp', 'ccm', 'continuity-check', 'service-group', 'nbar', 'protocol-pack', 'ethernet cfm', 'beacon interval', 'bfd interval', 'bfd neighbor', 'traceroute-cache', 'statistics-distribution', 'packet capture', 'schedule periodic']):
        return 'MNG'
    
    if any(kw in text for kw in ['crypto ca', 'certificate', 'cert-', 'radius', 'aaa ', 'pki', 'method-est', 'authentication md5']):
        return 'PKI'
    
    if any(kw in text for kw in ['interface ', 'ip address', 'encapsulation', 'frame-relay', 'dot1q', 'arp', 'nat64', 'bridge-domain', 'map-t', 'nptv6', 'ipv6 unicast-routing', 'nat66', 'ip vrf', 'ip nat inside', 'ip nat outside']):
        return 'INF'
    
    return 'OTHERS'

# Clasificar
df['category'] = df.apply(classify_intent, axis=1)

# Mostrar primeros 5 elementos de cada categoría
categories = df['category'].unique()

for category in sorted(categories):
    print(f"\n{'='*80}")
    print(f"CATEGORÍA: {category}")
    print(f"{'='*80}")
    
    samples = df[df['category'] == category].head(5)
    
    for idx, row in samples.iterrows():
        print(f"\n[{idx+1}] Question: {row['question']}")
        print(f"    Answer: {row['answer']}...")  # Primeros 200 caracteres
    
    print(f"\nTotal en {category}: {len(df[df['category'] == category])} elementos")

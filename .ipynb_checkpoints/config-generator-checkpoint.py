import requests
import json

# Configuración de Ollama
OLLAMA_API_URL = "http://localhost:11434/api/generate"
MODEL_ID = "llama3.1:8b"

def run_inference(requirement: str):
    """
    Envía una solicitud a Ollama con Llama 3.1 8B para clasificar el requerimiento
    """
    
    system_prompt = (
    "You are a network configuration assistant.\n"
    "Classify the requirement as one of: CP, RP, ACL, TN.\n"
    "CP is used for monitoring, performance, NetFlow, IP settings, and application layer configuration.\n"
    "RP is used for routing protocols such as OSPF, BGP, RIP, routing tables.\n"
    "ACL is used for access control lists and firewall rules.\n"
    "TN is used ONLY for tunnels and VPNs such as IPSec, GRE, or site-to-site tunnels.\n"
    "Return ONLY a valid JSON object in the following format:\n"
    "{\n"
    '  "type": "CP | RP | ACL | TN",\n'
    '  "steps": ["step 1", "step 2", "next steps"]\n'
    "}\n"
    "Rules:\n"
    "- Output ONLY JSON\n"
    "- No explanations\n"
    "- No markdown\n"
    "- No extra text"
 )

    
    prompt = f"{system_prompt}\n\nUser requirement: {requirement}"
    
    try:
        print(f"Conectando a Ollama en {OLLAMA_API_URL}...")
        print(f"Usando modelo: {MODEL_ID}")
        print("Generando respuesta...\n")
        
        response = requests.post(
            OLLAMA_API_URL,
            json={
                "model": MODEL_ID,
                "prompt": prompt,
                "stream": False,
                "temperature": 0.1
            },
            timeout=300  # 5 minutos de timeout
        )
        
        if response.status_code == 200:
            result = response.json()
            response_text = result.get("response", "")
            
            # Intentar parsear JSON
            try:
                response_json = json.loads(response_text)
                return response_json
            except json.JSONDecodeError:
                print("⚠️ El modelo no devolvió JSON válido:")
                print(response_text)
                return None
        else:
            print(f"❌ Error: {response.status_code}")
            print(response.text)
            return None
            
    except requests.exceptions.ConnectionError:
        print("❌ Error: No se puede conectar a Ollama.")
        print("Asegúrate de que Ollama esté corriendo en otra terminal:")
        print("  ollama serve")
        return None
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return None


def generate_cisco_config(requirement: str, low_level_steps: list, topology_info: str = ""):
    """
    Segunda fase: genera las configuraciones de Cisco IOS basadas en los pasos de bajo nivel
    """
    
    # Convertir los pasos en un string formateado
    steps_text = "\n".join([f"{i+1}. {step}" for i, step in enumerate(low_level_steps)])
    
    system_prompt = (
        "You are an expert network administrator that generates Cisco IOS commands.\n\n"
        
        "CRITICAL RULES:\n"
        "1. If the requirement is for VERIFICATION or TROUBLESHOOTING, use ONLY show/debug commands (show ip ospf neighbor, show running-config, debug, etc.)\n"
        "2. If the requirement is for CONFIGURATION, use config commands (router ospf, interface, ip address, etc.)\n"
        "3. Configure ONLY what is EXPLICITLY requested - DO NOT add extra commands, features, or configurations not mentioned\n"
        "4. DO NOT invent or assume ANY values: IPs, interfaces, hostnames, process IDs, subnet masks, VLANs, authentication, etc.\n"
        "5. DO NOT add authentication, costs, timers, priorities, or any feature NOT specifically requested\n"
        "6. If information is missing and you cannot complete the task, respond ONLY: <INSUFFICIENT_DATA: specify what is needed>\n"
        "7. Group ALL commands for each device under ONE separator: ~~~<device_name>~~~\n"
        "8. NO explanations, NO comments, NO markdown, ONLY commands\n"
        "9. If not applicable to configuration, respond ONLY: <No Configuration Requirements>\n\n"
        
        "OUTPUT FORMAT:\n"
        "~~~Device1~~~\n"
        "command1\n"
        "command2\n"
        "~~~Device2~~~\n"
        "command1\n"
        "command2\n"
    )
    
    user_prompt = f"Original requirement: {requirement}\n\nSteps to implement:\n{steps_text}"
    if topology_info:
        user_prompt += f"\n\nTopology information:\n{topology_info}"
    
    prompt = f"{system_prompt}\n\n{user_prompt}"
    
    try:
        print(f"\n🔧 Generando configuración de Cisco IOS...")
        
        response = requests.post(
            OLLAMA_API_URL,
            json={
                "model": MODEL_ID,
                "prompt": prompt,
                "stream": False,
                "temperature": 0.01
            },
            timeout=300
        )
        
        if response.status_code == 200:
            result = response.json()
            config_text = result.get("response", "")
            return config_text
        else:
            print(f"❌ Error en generación de config: {response.status_code}")
            print(response.text)
            return None
            
    except Exception as e:
        print(f"❌ Error generando configuración: {str(e)}")
        return None


if __name__ == "__main__":
    
    requirement = (
        "Configure OSPF process 10 area 0 on Router1 interface Gi0/0 with IP 192.168.1.1/30"
    )
    
    # Opcional: información de topología para evitar que el modelo asuma valores
    topology_info = (
        "Router1: hostname R1, interface GigabitEthernet0/0 with IP 192.168.1.1/30\n"
        "Router2: hostname R2, interface GigabitEthernet0/0 with IP 192.168.1.2/30\n"
        "OSPF process ID: 1\n"
        "OSPF area: 0"
    )
    
    # Si no tienes info de topología, déjalo vacío: topology_info = ""

    result = run_inference(requirement)

    if result:
        print("✅ Clasificación y pasos del modelo:")
        print(json.dumps(result, indent=2))
        
        # Segunda fase: generar configuración de Cisco IOS
        if "steps" in result and result["steps"]:
            print("\n" + "="*60)
            # Pasar topología vacía para probar si el modelo pide datos
            cisco_config = generate_cisco_config(requirement, result["steps"], topology_info="")
            
            if cisco_config:
                print("\n✅ Configuración de Cisco IOS generada:")
                print("="*60)
                print(cisco_config)
                print("="*60)
            else:
                print("❌ No se pudo generar la configuración de Cisco IOS")
        else:
            print("⚠️ No se encontraron pasos en la respuesta del modelo")
    else:
        print("❌ No se obtuvo respuesta válida del modelo")

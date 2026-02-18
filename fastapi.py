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


def generate_cisco_config(requirement: str, low_level_steps: list):
    """
    Segunda fase: genera las configuraciones de Cisco IOS basadas en los pasos de bajo nivel
    """
    
    # Convertir los pasos en un string formateado
    steps_text = "\n".join([f"{i+1}. {step}" for i, step in enumerate(low_level_steps)])
    
    system_prompt = (
        "You are an expert network administrator that need to create configs which are Cisco IOS configuration files to satisfy user low_level_description to identify required network values as IPs, interfaces and connections. "
        "You are not allowed to give any explanation of any type, you are allowed just to answer with the required commands to configure the device. "
        "Assume you are at global configuration mode in each device always. "
        "Separate each device configurations with the special identifier ~~~<device name>~~~. "
        "Finally you would ignore any other question that is not applicable to the configuration generation process just by answering <No Configuration Requirements>. "
        "Take in count you are not allowed to make assumptions of nothing, just use the user requirements and the topology description to achieve the requirement goal, if it is not possible to do it just with this data then you can assume the less as possible, but your primary goal is to solve the requirement without assuming anything. "
        "Don't make any explanation."
    )
    
    user_prompt = f"Original requirement: {requirement}\n\nLow level steps:\n{steps_text}"
    prompt = f"{system_prompt}\n\n{user_prompt}"
    
    try:
        print(f"\n🔧 Generando configuración de Cisco IOS...")
        
        response = requests.post(
            OLLAMA_API_URL,
            json={
                "model": MODEL_ID,
                "prompt": prompt,
                "stream": False,
                "temperature": 0.1
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
        "How can I verify and troubleshoot why OSPF neighbors are not forming between two Cisco routers?"

    )

    result = run_inference(requirement)

    if result:
        print("✅ Clasificación y pasos del modelo:")
        print(json.dumps(result, indent=2))
        
        # Segunda fase: generar configuración de Cisco IOS
        if "steps" in result and result["steps"]:
            print("\n" + "="*60)
            cisco_config = generate_cisco_config(requirement, result["steps"])
            
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

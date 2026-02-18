import requests
import json

# Configuración de Ollama
OLLAMA_API_URL = "http://localhost:11434/api/generate"
MODEL_ID = "llama3.1:8b"

def run_inference(requirement: str):
    """
    Envía una solicitud a Ollama con Llama 3.1 8B
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


if __name__ == "__main__":
    
    requirement = (
        "How can we monitor and store download time within the Cisco device "
        "and configure IP and application layer options for optimal performance?"
    )

    result = run_inference(requirement)

    if result:
        print("✅ Respuesta del modelo:")
        print(json.dumps(result, indent=2))
    else:
        print("❌ No se obtuvo respuesta válida del modelo")

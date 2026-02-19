from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import json
from typing import Optional

# Configuración de Ollama
OLLAMA_API_URL = "http://localhost:11434/api/generate"
MODEL_ID = "llama3.1:8b-instruct-q8_0"

app = FastAPI(title="Network Config Generator API")


class ConfigRequest(BaseModel):
    requirement: str
    network_state: Optional[str] = ""


class ConfigResponse(BaseModel):
    classification_type: str
    steps: list[str]
    cisco_config: str
    success: bool
    error_message: Optional[str] = None


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
            response_text = result.get("response", "")
            
            try:
                response_json = json.loads(response_text)
                return response_json
            except json.JSONDecodeError:
                return None
        else:
            return None
            
    except requests.exceptions.ConnectionError:
        raise HTTPException(status_code=503, detail="Cannot connect to Ollama. Make sure Ollama is running.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during inference: {str(e)}")


def generate_cisco_config(requirement: str, low_level_steps: list, topology_info: str = ""):
    """
    Segunda fase: genera las configuraciones de Cisco IOS basadas en los pasos de bajo nivel
    """
    
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
        user_prompt += f"\n\nNetwork state/topology:\n{topology_info}"
    
    prompt = f"{system_prompt}\n\n{user_prompt}"
    
    try:
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
            return None
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating config: {str(e)}")


@app.get("/")
def read_root():
    return {
        "message": "Network Config Generator API",
        "endpoints": {
            "/generate-config": "POST - Generate Cisco IOS configuration",
            "/health": "GET - Check API health"
        }
    }


@app.get("/health")
def health_check():
    """Check if Ollama is accessible"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            return {"status": "healthy", "ollama": "connected"}
        else:
            return {"status": "degraded", "ollama": "unreachable"}
    except:
        return {"status": "unhealthy", "ollama": "disconnected"}


@app.post("/generate-config", response_model=ConfigResponse)
def generate_config(request: ConfigRequest):
    """
    Generate Cisco IOS configuration based on requirement and network state
    
    Args:
        requirement: The user's network requirement in natural language
        network_state: Optional topology/network state information (IPs, interfaces, hostnames, etc.)
    
    Returns:
        ConfigResponse with classification, steps, and generated Cisco commands
    """
    
    # Fase 1: Clasificación y generación de pasos
    classification_result = run_inference(request.requirement)
    
    if not classification_result:
        raise HTTPException(
            status_code=500, 
            detail="Failed to classify requirement. Model did not return valid JSON."
        )
    
    if "type" not in classification_result or "steps" not in classification_result:
        raise HTTPException(
            status_code=500,
            detail="Invalid classification result format"
        )
    
    # Fase 2: Generación de configuración Cisco
    cisco_config = generate_cisco_config(
        request.requirement, 
        classification_result["steps"],
        request.network_state
    )
    
    if not cisco_config:
        raise HTTPException(
            status_code=500,
            detail="Failed to generate Cisco configuration"
        )
    
    return ConfigResponse(
        classification_type=classification_result["type"],
        steps=classification_result["steps"],
        cisco_config=cisco_config,
        success=True,
        error_message=None
    )


if __name__ == "__main__":
    import uvicorn
    print("🚀 Iniciando servidor FastAPI en http://localhost:8000")
    print("📝 Documentación interactiva en http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)

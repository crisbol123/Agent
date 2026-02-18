from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json

MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"

def run_inference(requirement: str):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You are a network configuration assistant.\n"
                "Classify the requirement as one of: CP, RP, ACL, TN.\n"
                "Return ONLY a valid JSON object in the following format:\n"
                "{\n"
                '  "type": "CP | RP | ACL | TN",\n'
                '  "steps": ["step 1", "step 2"]\n'
                "}\n"
                "Rules:\n"
                "- Output ONLY JSON\n"
                "- No explanations\n"
                "- No markdown\n"
                "- No extra text"
            )
        },
        {
            "role": "user",
            "content": requirement
        }
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt"
    ).to(model.device)

    outputs = model.generate(
        inputs,
        temperature=0.1,
        max_new_tokens=200
    )

    response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Validación básica de JSON (opcional pero recomendable)
    try:
        response_json = json.loads(response_text)
        return response_json
    except json.JSONDecodeError:
        print("⚠️ El modelo no devolvió JSON válido:")
        print(response_text)
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

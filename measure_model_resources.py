import argparse
import gc
import json
import os
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv


ENV_FILE = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=ENV_FILE)

HF_TOKEN = os.getenv("HF_TOKEN")
os.environ["HF_HOME"] = os.getenv("HF_HOME", r"D:\huggingface")
os.environ["HUGGINGFACE_HUB_CACHE"] = os.getenv("HUGGINGFACE_HUB_CACHE", r"D:\huggingface\hub")
os.environ.pop("TRANSFORMERS_CACHE", None)

HF_HUB_CACHE_DIR = os.environ["HUGGINGFACE_HUB_CACHE"]
os.makedirs(HF_HUB_CACHE_DIR, exist_ok=True)


MODELS = {
    "Llama-3.1-8B-Instruct": {
        "path": "meta-llama/Llama-3.1-8B-Instruct",
        "params": "8B",
    },
    "Zephyr-7B": {
        "path": "HuggingFaceH4/zephyr-7b-beta",
        "params": "7B",
    },
    "Qwen2.5-7B-Instruct": {
        "path": "Qwen/Qwen2.5-7B-Instruct",
        "params": "7B",
    },
    "Gemma-2-9B-it": {
        "path": "google/gemma-2-9b-it",
        "params": "9B",
    },
    "FLAN-T5-large": {
        "path": "google/flan-t5-large",
        "params": "780M",
        "architecture": "seq2seq",
    },
    "FLAN-T5-base": {
        "path": "google/flan-t5-base",
        "params": "250M",
        "architecture": "seq2seq",
    },
}


def bytes_to_gb(num_bytes: int) -> float:
    return round(float(num_bytes) / (1024 ** 3), 4)


def dir_size_bytes(path: str) -> int:
    total = 0
    for root, _, files in os.walk(path):
        for name in files:
            fpath = os.path.join(root, name)
            if os.path.isfile(fpath):
                total += os.path.getsize(fpath)
    return total


def get_model_cache_dir(cache_root: str, repo_id: str) -> str:
    # Estructura HF Hub para modelos: models--org--repo
    normalized = repo_id.strip().replace("/", "--")
    return os.path.join(cache_root, f"models--{normalized}")


def get_model_disk_usage_from_cache(cache_root: str, repo_id: str) -> tuple[int, str]:
    repo_cache_dir = get_model_cache_dir(cache_root, repo_id)
    if not os.path.isdir(repo_cache_dir):
        raise FileNotFoundError(
            f"No se encontro el modelo en cache local: {repo_cache_dir}"
        )

    # blobs contiene los pesos reales sin duplicados por snapshot.
    blobs_dir = os.path.join(repo_cache_dir, "blobs")
    if os.path.isdir(blobs_dir):
        return dir_size_bytes(blobs_dir), blobs_dir

    return dir_size_bytes(repo_cache_dir), repo_cache_dir


def load_models_from_json(models_file: str):
    with open(models_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError("El archivo JSON debe ser un objeto con formato {nombre_modelo: {path: ...}}")

    for model_name, cfg in data.items():
        if not isinstance(cfg, dict) or "path" not in cfg:
            raise ValueError(f"Modelo invalido en JSON: {model_name}. Debe incluir al menos 'path'.")

    return data


def load_model_like_evaluate_generation(model_cfg: dict, allow_download: bool):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    device = "cuda:0"
    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg["path"],
        trust_remote_code=model_cfg.get("trust_remote_code", False),
        cache_dir=HF_HUB_CACHE_DIR,
        token=HF_TOKEN,
        local_files_only=not allow_download,
    )
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_cfg["path"],
            torch_dtype=torch.bfloat16,
            device_map=device,
            trust_remote_code=model_cfg.get("trust_remote_code", False),
            cache_dir=HF_HUB_CACHE_DIR,
            token=HF_TOKEN,
            local_files_only=not allow_download,
        )
    except Exception as e:
        if "Invalid device argument" not in str(e):
            raise
        model = AutoModelForCausalLM.from_pretrained(
            model_cfg["path"],
            torch_dtype=torch.bfloat16,
            trust_remote_code=model_cfg.get("trust_remote_code", False),
            cache_dir=HF_HUB_CACHE_DIR,
            token=HF_TOKEN,
            local_files_only=not allow_download,
        )
        model.to(device)
    return tokenizer, model


def load_model_seq2seq_like_evaluate_generation(model_cfg: dict, allow_download: bool):
    import torch
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    device = "cuda:0"
    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg["path"],
        trust_remote_code=model_cfg.get("trust_remote_code", False),
        cache_dir=HF_HUB_CACHE_DIR,
        token=HF_TOKEN,
        local_files_only=not allow_download,
    )
    try:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_cfg["path"],
            torch_dtype=torch.bfloat16,
            device_map=device,
            trust_remote_code=model_cfg.get("trust_remote_code", False),
            cache_dir=HF_HUB_CACHE_DIR,
            token=HF_TOKEN,
            local_files_only=not allow_download,
        )
    except Exception as e:
        if "Invalid device argument" not in str(e):
            raise
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_cfg["path"],
            torch_dtype=torch.bfloat16,
            trust_remote_code=model_cfg.get("trust_remote_code", False),
            cache_dir=HF_HUB_CACHE_DIR,
            token=HF_TOKEN,
            local_files_only=not allow_download,
        )
        model.to(device)
    return tokenizer, model


def measure_model(model_name: str, model_cfg: dict, allow_download: bool = False):
    from huggingface_hub import snapshot_download
    import torch

    result = {
        "model_name": model_name,
        "path": model_cfg["path"],
        "params": model_cfg.get("params", "N/A"),
        "cache_repo_path": None,
        "snapshot_path": None,
        "disk_size_bytes": None,
        "disk_size_gb": None,
        "vram_allocated_after_load_bytes": None,
        "vram_allocated_after_load_gb": None,
        "vram_peak_during_load_bytes": None,
        "vram_peak_during_load_gb": None,
        "status": "ok",
        "error": None,
    }

    t0 = time.time()

    try:
        disk_bytes, cache_size_source = get_model_disk_usage_from_cache(
            HF_HUB_CACHE_DIR,
            model_cfg["path"],
        )
        result["cache_repo_path"] = cache_size_source
        result["disk_size_bytes"] = int(disk_bytes)
        result["disk_size_gb"] = bytes_to_gb(disk_bytes)

        snapshot_path = snapshot_download(
            repo_id=model_cfg["path"],
            cache_dir=HF_HUB_CACHE_DIR,
            token=HF_TOKEN,
            local_files_only=not allow_download,
            resume_download=allow_download,
        )
        result["snapshot_path"] = snapshot_path
    except Exception as e:
        result["status"] = "error"
        result["error"] = f"Error descargando/calculando espacio en disco: {e}"
        result["elapsed_s"] = round(time.time() - t0, 2)
        return result

    if not torch.cuda.is_available():
        result["status"] = "warning"
        result["error"] = "CUDA no disponible. Solo se reporta espacio en disco."
        result["elapsed_s"] = round(time.time() - t0, 2)
        return result

    tokenizer = None
    model = None
    try:
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        architecture = model_cfg.get("architecture", "causal")
        if architecture == "seq2seq":
            tokenizer, model = load_model_seq2seq_like_evaluate_generation(
                model_cfg,
                allow_download,
            )
        else:
            tokenizer, model = load_model_like_evaluate_generation(
                model_cfg,
                allow_download,
            )

        allocated = torch.cuda.memory_allocated(0)
        peak = torch.cuda.max_memory_allocated(0)

        result["vram_allocated_after_load_bytes"] = int(allocated)
        result["vram_allocated_after_load_gb"] = bytes_to_gb(allocated)
        result["vram_peak_during_load_bytes"] = int(peak)
        result["vram_peak_during_load_gb"] = bytes_to_gb(peak)

    except Exception as e:
        result["status"] = "error"
        result["error"] = f"Error cargando modelo en GPU: {e}"
    finally:
        if model is not None:
            del model
        if tokenizer is not None:
            del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    result["elapsed_s"] = round(time.time() - t0, 2)
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Mide espacio en disco y VRAM de modelos Hugging Face cargandolos uno por uno."
    )
    parser.add_argument(
        "--models-file",
        type=str,
        default=None,
        help="JSON opcional con modelos en formato {nombre: {path: ..., params: ...}}",
    )
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Permite descargar desde Hugging Face si faltan archivos en cache. Por defecto: solo local.",
    )
    args = parser.parse_args()

    models = MODELS
    if args.models_file:
        models = load_models_from_json(args.models_file)

    print("=" * 80)
    print("AUDITORIA DE ESPACIO Y VRAM POR MODELO")
    print(f"Modelos a medir: {len(models)}")
    print(f"Cache HF: {HF_HUB_CACHE_DIR}")
    print(f"Modo: {'con descarga permitida' if args.allow_download else 'solo cache local'}")
    print("=" * 80)

    results = []
    for model_name, model_cfg in models.items():
        print(f"\n[{model_name}] Descargando/cargando...")
        r = measure_model(model_name, model_cfg, allow_download=args.allow_download)
        results.append(r)

        disk_txt = f"{r['disk_size_gb']} GB" if r["disk_size_gb"] is not None else "N/A"
        vram_txt = f"{r['vram_peak_during_load_gb']} GB" if r["vram_peak_during_load_gb"] is not None else "N/A"
        print(f"  Espacio disco: {disk_txt}")
        print(f"  VRAM pico:     {vram_txt}")
        print(f"  Estado:        {r['status']}")
        if r["error"]:
            print(f"  Detalle:       {r['error']}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_file = f"model_resource_report_{timestamp}.json"
    csv_file = f"model_resource_report_{timestamp}.csv"

    output = {
        "timestamp": timestamp,
        "cache_dir": HF_HUB_CACHE_DIR,
        "models_evaluated": list(models.keys()),
        "results": results,
    }

    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    pd.DataFrame(results).to_csv(csv_file, index=False, encoding="utf-8")

    print("\n" + "=" * 80)
    print(f"Reporte JSON: {json_file}")
    print(f"Reporte CSV:  {csv_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()
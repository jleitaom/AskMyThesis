"""
Load run configuration from a YAML file.

Every script imports CFG from here instead of hard-coding settings. Which file is used:
  1. --config path/to/file.yaml on the command line
     (for Streamlit: `streamlit run app.py -- --config path/to/file.yaml`)
  2. the ASKMYTHESIS_CONFIG env var
  3. config.yaml at the repo root

Keys missing from the file fall back to DEFAULTS below, so a variant config only needs
the keys it changes. Unknown keys raise, so a typo can't silently fall back to a default.
Relative paths under `paths:` are resolved against the repo root.
"""

# IMPORTS ---------------------------------------------------------------------------------

import argparse
import copy
import os
from pathlib import Path

import yaml

# CONFIGURATION --------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = ROOT / "config.yaml"

DEFAULTS = {
    "paths": {
        "pdf": "data/raw/thesis.pdf",
        "extracted": "data/processed/extracted_sections.json",
        "cleaned": "data/processed/cleaned_sections.json",
        "chunks": "data/processed/chunks.json",
        "chroma_dir": "data/chroma",
        "golden": "data/golden/golden.json",
        "retrieval_reports": "evals/retrieval",
        "generation_reports": "evals/generation",
    },
    "extraction": {
        "page_start": 20,
        "page_end": 137,
    },
    "embedding": {
        "model": "BAAI/bge-m3",
    },
    "chunking": {
        "chunk_size": 500,
        "chunk_overlap": 75,
    },
    "index": {
        "collection": "thesis",
    },
    "retrieval": {
        "k": 4,
        "search_type": "similarity",
        "fetch_k": 20,
    },
    "generation": {
        "backend": "ollama",
        "ollama_model": "qwen2.5:7b",
        "hf_model": "Qwen/Qwen2.5-7B-Instruct",
        "max_new_tokens": 512,
        "temperature": 0,
    },
    "eval": {
        "retrieval": {
            "k_values": [1, 2, 3, 4, 5, 6, 7, 10],
            "search_types": ["similarity", "mmr"],
        },
        "generation": {
            "judge_model": "gemma3:12b",
            "judge_num_predict": 4096,
            "use_cache": True,
            "sample": None,
            "ragas_sample": None,
            "ragas_max_workers": 2,
            "ragas_timeout": 600,
            "ragas_metrics": ["faithfulness", "answer_relevancy", "semantic_similarity"],
        },
    },
}

# UTILITY FUNCTIONS -----------------------------------------------------------------------

def _merge(defaults, overrides, prefix=""):
    """
    Recursively overlay overrides onto a copy of defaults, rejecting unknown keys.
    """
    merged = copy.deepcopy(defaults)

    for key, value in (overrides or {}).items():
        name = f"{prefix}{key}"
        if key not in defaults:
            raise KeyError(f"Unknown config key {name!r}")
        if isinstance(defaults[key], dict):
            if not isinstance(value, dict):
                raise TypeError(f"Config key {name!r} must be a mapping")
            merged[key] = _merge(defaults[key], value, prefix=f"{name}.")
        else:
            merged[key] = value

    return merged


def _config_path():
    """
    Resolve which YAML file to load: --config flag, then env var, then repo default.
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config")
    args, _ = parser.parse_known_args()

    return Path(args.config or os.getenv("ASKMYTHESIS_CONFIG") or DEFAULT_CONFIG_PATH)


def load_config(path=None):
    """
    Load the YAML config, fill in defaults, and resolve paths against the repo root.
    """
    path = Path(path) if path else _config_path()
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    cfg = _merge(DEFAULTS, yaml.safe_load(path.read_text(encoding="utf-8")))
    cfg["paths"] = {name: ROOT / p for name, p in cfg["paths"].items()}
    cfg["source"] = path

    return cfg


CFG = load_config()

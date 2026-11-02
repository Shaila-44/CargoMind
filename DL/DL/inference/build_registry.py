"""Regenerate dl/inference/model_registry.json from the existing result tables (no training).

    .venv/bin/python -m dl.inference.build_registry
"""
import json

from dl.inference.registry import REGISTRY_PATH, build

if __name__ == "__main__":
    reg = build()
    REGISTRY_PATH.write_text(json.dumps(reg, indent=2))
    for k, v in reg["models"].items():
        print(f"{k:30s} seed {v['seed']}  val {v['val_rmse_h_this_seed']:.4f} h  {v['checkpoint']}")

#!/usr/bin/env python3
"""
Add a new network to the GitHub Pages site.

Usage:
    python add_network.py <slug> <notebook_path> <weights_path> [--name "Name"] [--description "..."]

Example:
    python add_network.py fashion-mnist ./Fashion-MNIST/notebook.ipynb ./Fashion-MNIST/weights.npz \
        --name "Fashion-MNIST" --description "Classifies clothing items from the Fashion-MNIST dataset"

This will:
  1. Create docs/networks/<slug>/
  2. Copy the notebook
  3. Convert weights.npz → weights.json
  4. Generate a template info.json (edit it to fill in details)
  5. Add the network to docs/networks/registry.json
"""

import argparse
import json
import os
import shutil
import sys

import numpy as np


DOCS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "docs")
REGISTRY_PATH = os.path.join(DOCS_DIR, "networks", "registry.json")


def convert_weights(npz_path):
    """Load weights.npz and return a JSON-serialisable dict."""
    data = np.load(npz_path)
    weights = {}
    for key in data.files:
        weights[key] = data[key].tolist()
    print(f"  Loaded {len(data.files)} weight arrays: {', '.join(data.files)}")
    return weights


def detect_layers(weights):
    """Try to auto-detect layer structure from weight key names."""
    layers = []
    # Look for keys like weights_0_1, weights_1_2, etc.
    weight_keys = sorted([k for k in weights if k.startswith("weights_")])
    bias_keys = {k for k in weights if k.startswith("bias_") or k.startswith("biases_")}

    for i, wk in enumerate(weight_keys):
        # Try to find a matching bias key
        parts = wk.replace("weights_", "")
        bias_key = None
        for prefix in ("bias_", "biases_"):
            candidate = prefix + parts
            if candidate in bias_keys:
                bias_key = candidate
                break

        # Default activation: softmax for last layer, tanh otherwise
        is_last = (i == len(weight_keys) - 1)
        activation = "softmax" if is_last else "tanh"

        layer = {"weights": wk, "bias": bias_key, "activation": activation}
        layers.append(layer)

    if not layers:
        # Fallback: just list all keys as layers
        for k in sorted(weights.keys()):
            layers.append({"weights": k, "bias": None, "activation": "linear"})

    return layers


def detect_architecture(weights, layers):
    """Build an architecture string like '784 → 100 (tanh) → 10 (softmax)'."""
    parts = []
    first_key = layers[0]["weights"]
    input_size = len(weights[first_key])
    parts.append(str(input_size))

    for layer in layers:
        w = weights[layer["weights"]]
        output_size = len(w[0]) if isinstance(w[0], list) else 1
        parts.append(f"{output_size} ({layer['activation']})")

    return " → ".join(parts)


def main():
    parser = argparse.ArgumentParser(description="Add a network to the site")
    parser.add_argument("slug", help="URL-safe identifier (e.g. 'fashion-mnist')")
    parser.add_argument("notebook", help="Path to the .ipynb notebook file")
    parser.add_argument("weights", help="Path to the weights.npz file")
    parser.add_argument("--name", default=None, help="Display name (default: slug titlecased)")
    parser.add_argument("--description", default="", help="Short description")
    parser.add_argument("--input-type", default="mnist", help="Input type for the try-it widget (default: mnist)")
    args = parser.parse_args()

    slug = args.slug
    name = args.name or slug.replace("-", " ").title()
    net_dir = os.path.join(DOCS_DIR, "networks", slug)

    # Check inputs exist
    if not os.path.isfile(args.notebook):
        print(f"Error: notebook not found: {args.notebook}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(args.weights):
        print(f"Error: weights not found: {args.weights}", file=sys.stderr)
        sys.exit(1)

    # Create network directory
    os.makedirs(net_dir, exist_ok=True)
    print(f"Network directory: {net_dir}")

    # Copy notebook
    nb_dest = os.path.join(net_dir, "notebook.ipynb")
    shutil.copy2(args.notebook, nb_dest)
    print(f"  Copied notebook → {nb_dest}")

    # Convert weights
    print("  Converting weights.npz → weights.json ...")
    weights = convert_weights(args.weights)
    weights_dest = os.path.join(net_dir, "weights.json")
    with open(weights_dest, "w") as f:
        json.dump(weights, f)
    print(f"  Wrote {os.path.getsize(weights_dest) / 1024:.0f} KB → {weights_dest}")

    # Detect layers and architecture
    layers = detect_layers(weights)
    architecture = detect_architecture(weights, layers)
    print(f"  Detected architecture: {architecture}")
    print(f"  Detected {len(layers)} layers")

    # Create info.json
    info = {
        "name": name,
        "description": args.description,
        "architecture": architecture,
        "techniques": [],
        "results": {
            "test_accuracy": "TODO",
            "summary": "TODO — fill in test results"
        },
        "layers": layers,
        "weights": f"networks/{slug}/weights.json",
        "notebook": f"networks/{slug}/notebook.ipynb",
        "notes": None,
        "input_size": len(weights[layers[0]["weights"]]),
        "input_type": args.input_type
    }
    info_path = os.path.join(net_dir, "info.json")
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)
    print(f"  Created {info_path}")

    # Update registry
    if os.path.isfile(REGISTRY_PATH):
        with open(REGISTRY_PATH) as f:
            registry = json.load(f)
    else:
        registry = []

    # Don't add duplicates
    if not any(e["id"] == slug for e in registry):
        registry.append({
            "id": slug,
            "path": f"networks/{slug}/info.json"
        })
        with open(REGISTRY_PATH, "w") as f:
            json.dump(registry, f, indent=2)
        print(f"  Added '{slug}' to registry ({len(registry)} networks total)")
    else:
        print(f"  '{slug}' already in registry — skipped")

    print()
    print(f"Done! Next steps:")
    print(f"  1. Edit {info_path} to fill in techniques, results, etc.")
    print(f"  2. Optionally add a notes.md in {net_dir}/")
    print(f"  3. Commit and push to deploy")


if __name__ == "__main__":
    main()

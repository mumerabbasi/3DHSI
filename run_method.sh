#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PIPELINE_PYTHON:-python}"

if [[ $# -ne 1 || "$1" == "--help" || "$1" == "-h" ]]; then
    echo "Usage: $0 interaction_XX" >&2
    exit 2
fi

INTERACTION="$1"
if [[ ! -f "$ROOT/01_Generate_SIG/input_prompts/$INTERACTION/input_scene.json" ]]; then
    echo "Missing input scene for $INTERACTION" >&2
    exit 1
fi

cd "$ROOT"

echo "[1/5] Generate SIG: $INTERACTION"
"$PYTHON_BIN" 01_Generate_SIG/01_generate_sig.py --interaction_name "$INTERACTION"

echo "[2/5] Generate human frame: $INTERACTION"
"$PYTHON_BIN" 02_Generate_Human_Frame/01_generate_human_frame.py --interaction_name "$INTERACTION" --overwrite

echo "[3/5] Estimate contact: $INTERACTION"
"$PYTHON_BIN" 03_Estimate_Contact_Agentic/01_estimate_agentic_contact.py --interaction_name "$INTERACTION" --overwrite

echo "[4/5] Estimate human pose: $INTERACTION"
"$PYTHON_BIN" 04_Estimate_Human_Pose/01_estimate_static_pose.py --interaction_name "$INTERACTION"

echo "[5/5] Optimize static scene: $INTERACTION"
"$PYTHON_BIN" 05_Optimize_Static_Scene/01_optimize_static_scene.py --interaction_name "$INTERACTION"

echo "Optimized output: $ROOT/05_Optimize_Static_Scene/output/$INTERACTION"

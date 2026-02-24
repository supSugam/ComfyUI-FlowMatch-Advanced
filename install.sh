#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 {COMFYUI_DIR}" >&2
}

if [[ $# -ne 1 ]]; then
  usage
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INPUT_DIR="$1"

if [[ ! -d "$INPUT_DIR" ]]; then
  echo "Error: path does not exist: $INPUT_DIR" >&2
  exit 1
fi

if [[ -d "$INPUT_DIR/custom_nodes" ]]; then
  CUSTOM_NODES_DIR="$INPUT_DIR/custom_nodes"
elif [[ "$(basename "$INPUT_DIR")" == "custom_nodes" ]]; then
  CUSTOM_NODES_DIR="$INPUT_DIR"
else
  echo "Error: could not find custom_nodes in: $INPUT_DIR" >&2
  echo "Pass your ComfyUI root dir or the custom_nodes dir." >&2
  exit 1
fi

NODE_NAME="$(basename "$SCRIPT_DIR")"
DEST_DIR="$CUSTOM_NODES_DIR/$NODE_NAME"

mkdir -p "$CUSTOM_NODES_DIR"

# Remove case-variant installs of the same node name to avoid duplicates.
NODE_NAME_LC="$(printf '%s' "$NODE_NAME" | tr '[:upper:]' '[:lower:]')"
for existing_dir in "$CUSTOM_NODES_DIR"/*; do
  [[ -d "$existing_dir" ]] || continue
  existing_name="$(basename "$existing_dir")"
  existing_name_lc="$(printf '%s' "$existing_name" | tr '[:upper:]' '[:lower:]')"
  if [[ "$existing_name_lc" == "$NODE_NAME_LC" && "$existing_dir" != "$DEST_DIR" ]]; then
    rm -rf "$existing_dir"
  fi
done

# If already installed, wipe and replace to guarantee overwrite behavior.
if [[ -d "$DEST_DIR" ]]; then
  rm -rf "$DEST_DIR"
fi
mkdir -p "$DEST_DIR"

cp -f "$SCRIPT_DIR/__init__.py" "$DEST_DIR/"
cp -f "$SCRIPT_DIR/flowmatch_nodes.py" "$DEST_DIR/"
if [[ -f "$SCRIPT_DIR/README.md" ]]; then
  cp -f "$SCRIPT_DIR/README.md" "$DEST_DIR/"
fi

echo "Installed to: $DEST_DIR"
echo "Restart ComfyUI to load the updated nodes."

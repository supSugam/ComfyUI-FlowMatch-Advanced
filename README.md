# ComfyUI-FlowMatch-Advanced

Advanced Flow Matching and Noise Schedule scaling for DiT (Diffusion Transformer) models. This node replicates the precise inference logic from the `lora-scripts` ecosystem for **Flux**, **Z-Image**, and **Qwen-Image**.

## Overview

Modern DiT models like Flux and Qwen use **Rectified Flow** or **Discrete Flow Matching**. A critical component of their success is the "Time Shift" logic, which adjusts the noise schedule based on the image's sequence length (resolution). 

Without this shift, images at higher resolutions (e.g., 2048px) often suffer from structural collapse or lack of detail because the noise schedule is too "steep." This node automates that math for you.

## Features

- **Resolution-Aware Shifting:** Automatically calculates the `mu` value based on your target resolution, replicating the $0.5 
ightarrow 1.15$ shift range used in Black Forest Labs' Flux and Lumina-Next.
- **DiT Optimized Presets:** Specialized math for:
  - **Flux:** MM-DiT sequence scaling.
  - **Z-Image:** (Lumina-Next/Flag-DiT) Discrete flow scaling.
  - **Qwen-Image:** (Lumina-Qwen) LLM-backbone specific shifting.
- **V-Prediction Enforcement:** Automatically patches the model to `v_prediction` mode to prevent washed-out/grey images.
- **Universal Compatibility:** Works with both `Load Checkpoint` and `Load Diffusion Model` workflows.

## Installation

1. Navigate to your `ComfyUI/custom_nodes/` directory.
2. Clone this repository:
   ```bash
   git clone https://github.com/supSugam/ComfyUI-FlowMatch-Advanced
   ```
3. Restart ComfyUI.

## Usage

### Node Parameters

- **model**: The DiT model you want to patch (from Load Checkpoint or Load Diffusion Model).
- **model_type**: Choose the architecture (`flux`, `z-image`, or `qwen`).
- **shift**: The manual shift value (used if `resolution_aware` is off, or as the max boundary for Qwen).
- **resolution_aware**: (Recommended) When enabled, it ignores the manual shift and calculates the optimal trajectory based on width/height.
- **width/height**: Your target generation dimensions (used for the `mu` calculation).

### Recommended Workflow

1. **Load Model:** Load your Flux/Z-Image/Qwen model.
2. **Apply FlowMatch:** Connect the **MODEL** output to the **DiT FlowMatch** node.
3. **Connect Sampler:** Connect the patched **MODEL** to your `KSampler` or `SamplerCustom`.
4. **Sampler Settings:**
   - **Sampler:** `euler`
   - **Scheduler:** `normal` or `sgm_uniform`
   - **Denoise:** `1.0` (for full generation)

## Why this node?

In the original `lora-scripts` and official inference implementations, the noise schedule isn't static. It follows this formula:

$$ \sigma = \frac{shift \cdot t}{1 + (shift - 1) \cdot t} $$

And for Flux, the $shift$ ($mu$) is determined by:
$$ mu = lerp(0.5, 1.15, \text{sequence\_length}) $$

This node brings that exact mathematical precision to ComfyUI, ensuring your images look exactly as they would when generated with the specialized training scripts.

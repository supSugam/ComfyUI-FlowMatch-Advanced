# ComfyUI-FlowMatch-Advanced

Custom nodes to make ComfyUI sampling closer to `ai-toolkit` flow-matching behavior for **Flux**, **Qwen-Image**, and **Z-Image**.

## Nodes

- `FlowMatch Sigmas (ai-toolkit style)`  
  Outputs `SIGMAS` for `SamplerCustomAdvanced` using ai-toolkit-style formulas and defaults.
- `DiT FlowMatch Model (ai-toolkit style)`  
  Patches `model_sampling` with a real Comfy sampling object (Comfy-native patching), not a callback.

## Model Presets

- `flux`: dynamic shift (`base_shift=0.5`, `max_shift=1.15`, `max_seq_len=4096`)
- `qwen`: dynamic shift (`base_shift=0.5`, `max_shift=0.9`, `max_seq_len=8192`)
- `z-image`: static shift (`shift=3.0`)

## Recommended Workflow (Closest to ai-toolkit)

1. Load your model and LoRA.
2. Use `DiT FlowMatch Model (ai-toolkit style)` on the model (set `model_type=auto` unless needed).
3. Use `FlowMatch Sigmas (ai-toolkit style)` to generate `SIGMAS`.
4. Run `SamplerCustomAdvanced`:
   - sampler: `euler` (or `res_multistep` if that is your trained setup)
   - sigmas: from `FlowMatch Sigmas (ai-toolkit style)`
   - guider: `CFGGuider` (typically `cfg=1.0` for flow models)
5. Decode with VAE.

## Notes

- This is designed for the `SamplerCustomAdvanced` path.  
  `KSampler` cannot directly inject this exact custom sigma schedule.
- `force_aitk_timesteps=true` uses `1.0 -> 1.0/steps` timesteps before shift math, matching ai-toolkit behavior more closely.
- Default `steps=25` is set for closer ai-toolkit-style sample quality (you can still drop to 20 for faster Flux previews).

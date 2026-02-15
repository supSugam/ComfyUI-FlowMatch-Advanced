import torch
import math

class ModelSamplingDiTFlow:
    """
    Advanced Flow Matching Scaling for DiT models (Flux, Z-Image, Qwen).
    Replicates the exact logic from lora-scripts for seamless ComfyUI integration.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "model_type": (["flux", "z-image", "qwen"], {"default": "flux"}),
                "shift": ("FLOAT", {"default": 1.15, "min": 0.0, "max": 10.0, "step": 0.01}),
                "resolution_aware": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "width": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 8}),
                "height": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 8}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply_dit_flow"
    CATEGORY = "sampling/custom_sampling"

    def apply_dit_flow(self, model, model_type, shift, resolution_aware, width=1024, height=1024):
        m = model.clone()

        def get_lin_function(x1, y1, x2, y2):
            m_val = (y2 - y1) / (x2 - x1)
            b_val = y1 - m_val * x1
            return lambda x: m_val * x + b_val

        def time_shift(mu, t):
            return torch.exp(torch.tensor(mu)) / (torch.exp(torch.tensor(mu)) + (1 / t - 1) ** 1.0)

        def patch_sampling(filter_out, sigma_data):
            t = sigma_data["timesteps"]
            if t.max() > 10.0:
                t = t / 1000.0

            if resolution_aware:
                # Calculate sequence length based on packed latents (typical for Flux/DiT)
                # Flux uses 16x16 patch packing, so seq_len = (w/16) * (h/16)
                # We use the provided width/height to calculate the mu shift
                seq_len = (width // 16) * (height // 16)
                
                if model_type == "flux":
                    # Flux standard: linear interpolation between 0.5 and 1.15
                    mu = get_lin_function(256, 0.5, 4096, 1.15)(seq_len)
                elif model_type == "qwen":
                    # Qwen/Lumina often uses a slightly steeper shift for LLM-based embeddings
                    mu = get_lin_function(256, 0.5, 4096, shift)(seq_len)
                else: # z-image
                    mu = shift
            else:
                mu = shift

            # Calculate new sigmas
            sigmas = time_shift(mu, t)
            return sigmas

        # Apply patches
        m.add_object_patch("model_sampling", patch_sampling)
        
        # All these models are Flow-based and require v_prediction
        m.add_object_patch("parameterization", "v_prediction")
        
        return (m,)

NODE_CLASS_MAPPINGS = { "ModelSamplingDiTFlow": ModelSamplingDiTFlow }
NODE_DISPLAY_NAME_MAPPINGS = { "ModelSamplingDiTFlow": "DiT FlowMatch (Flux/Z-Image/Qwen)" }

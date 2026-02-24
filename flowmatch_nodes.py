import math

import comfy.model_sampling
import torch


AITK_FLOWMATCH_CONFIG = {
    "flux": {
        "base_image_seq_len": 256,
        "base_shift": 0.5,
        "max_image_seq_len": 4096,
        "max_shift": 1.15,
        "shift": 3.0,
        "use_dynamic_shifting": True,
    },
    "qwen": {
        "base_image_seq_len": 256,
        "base_shift": 0.5,
        "max_image_seq_len": 8192,
        "max_shift": 0.9,
        "shift": 1.0,
        "use_dynamic_shifting": True,
    },
    "z-image": {
        "shift": 3.0,
        "use_dynamic_shifting": False,
    },
}


def _calculate_shift(image_seq_len, base_seq_len, max_seq_len, base_shift, max_shift):
    m_val = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b_val = base_shift - m_val * base_seq_len
    return image_seq_len * m_val + b_val


def _dynamic_time_shift(mu, t):
    t = torch.clamp(t, min=1e-6, max=1.0)
    exp_mu = math.exp(float(mu))
    return exp_mu / (exp_mu + (1.0 / t - 1.0))


def _static_time_shift(shift_value, t):
    t = torch.clamp(t, min=1e-6, max=1.0)
    return (shift_value * t) / (1.0 + (shift_value - 1.0) * t)


def _resolve_model_type(model, model_type):
    if model_type != "auto":
        return model_type

    try:
        image_model = str(model.model.model_config.unet_config.get("image_model", "")).lower()
    except Exception:
        image_model = ""

    if image_model == "qwen_image":
        return "qwen"
    if image_model == "lumina2":
        return "z-image"
    if "flux" in image_model:
        return "flux"
    return "flux"


def _get_effective_shift(model_type, shift, resolution_aware, width, height):
    config = AITK_FLOWMATCH_CONFIG[model_type]
    if resolution_aware and config.get("use_dynamic_shifting", False):
        seq_len = (width // 16) * (height // 16)
        return _calculate_shift(
            seq_len,
            config["base_image_seq_len"],
            config["max_image_seq_len"],
            config["base_shift"],
            config["max_shift"],
        )
    if shift > 0.0:
        return shift
    return config["shift"]


class ModelSamplingDiTFlow:
    """
    ai-toolkit style FlowMatch model patch for DiT models.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "model_type": (["auto", "flux", "z-image", "qwen"], {"default": "auto"}),
                "shift": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "resolution_aware": ("BOOLEAN", {"default": True}),
                "width": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 8}),
                "height": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 8}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply_dit_flow"
    CATEGORY = "advanced/model"

    def apply_dit_flow(self, model, model_type, shift, resolution_aware, width, height):
        resolved_model_type = _resolve_model_type(model, model_type)
        effective_shift = _get_effective_shift(resolved_model_type, shift, resolution_aware, width, height)
        m = model.clone()

        if resolved_model_type in ("flux", "qwen") and resolution_aware:
            sampling_base = comfy.model_sampling.ModelSamplingFlux
            sampling_kwargs = {"shift": effective_shift}
        else:
            sampling_base = comfy.model_sampling.ModelSamplingDiscreteFlow
            sampling_kwargs = {"shift": effective_shift, "multiplier": 1.0}

        class ModelSamplingAdvanced(sampling_base, comfy.model_sampling.CONST):
            pass

        model_sampling = ModelSamplingAdvanced(model.model.model_config)
        model_sampling.set_parameters(**sampling_kwargs)
        m.add_object_patch("model_sampling", model_sampling)
        return (m,)


class AIToolkitFlowMatchScheduler:
    """
    ai-toolkit style SIGMAS output for SamplerCustomAdvanced.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_type": (["auto", "flux", "z-image", "qwen"], {"default": "auto"}),
                "steps": ("INT", {"default": 25, "min": 1, "max": 10000}),
                "width": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 8}),
                "height": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 8}),
                "shift": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "resolution_aware": ("BOOLEAN", {"default": True}),
                "force_aitk_timesteps": ("BOOLEAN", {"default": True}),
                "append_zero": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "model": ("MODEL",),
            },
        }

    RETURN_TYPES = ("SIGMAS",)
    FUNCTION = "get_sigmas"
    CATEGORY = "sampling/custom_sampling/schedulers"

    def get_sigmas(
        self,
        model_type,
        steps,
        width,
        height,
        shift,
        resolution_aware,
        force_aitk_timesteps,
        append_zero,
        model=None,
    ):
        if model is not None:
            resolved_model_type = _resolve_model_type(model, model_type)
        else:
            resolved_model_type = "flux" if model_type == "auto" else model_type

        config = AITK_FLOWMATCH_CONFIG[resolved_model_type]
        steps = max(int(steps), 1)

        if force_aitk_timesteps:
            t = torch.linspace(1.0, 1.0 / float(steps), steps, dtype=torch.float32)
        else:
            t = torch.linspace(1.0, 0.0, steps + 1, dtype=torch.float32)[:-1]

        if resolution_aware and config.get("use_dynamic_shifting", False):
            seq_len = (width // 16) * (height // 16)
            mu = _calculate_shift(
                seq_len,
                config["base_image_seq_len"],
                config["max_image_seq_len"],
                config["base_shift"],
                config["max_shift"],
            )
            sigmas = _dynamic_time_shift(mu, t)
        else:
            effective_shift = shift if shift > 0.0 else config["shift"]
            sigmas = _static_time_shift(effective_shift, t)

        sigmas = sigmas.to(dtype=torch.float32)
        if append_zero:
            sigmas = torch.cat([sigmas, sigmas.new_zeros((1,))])

        return (sigmas,)


NODE_CLASS_MAPPINGS = {
    "ModelSamplingDiTFlow": ModelSamplingDiTFlow,
    "AIToolkitFlowMatchScheduler": AIToolkitFlowMatchScheduler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ModelSamplingDiTFlow": "DiT FlowMatch Model (ai-toolkit style)",
    "AIToolkitFlowMatchScheduler": "FlowMatch Sigmas (ai-toolkit style)",
}

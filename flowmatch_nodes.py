import math
import os
import sys

import comfy.model_management
import comfy.model_sampling
import comfy.nested_tensor
import comfy.sample
import comfy.samplers
import comfy.utils
import latent_preview
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

AITK_SCHEDULER_CONFIG = {
    "flux": {
        "base_image_seq_len": 256,
        "base_shift": 0.5,
        "max_image_seq_len": 4096,
        "max_shift": 1.15,
        "num_train_timesteps": 1000,
        "shift": 3.0,
        "use_dynamic_shifting": True,
    },
    "qwen": {
        "base_image_seq_len": 256,
        "base_shift": 0.5,
        "invert_sigmas": False,
        "max_image_seq_len": 8192,
        "max_shift": 0.9,
        "num_train_timesteps": 1000,
        "shift": 1.0,
        "shift_terminal": 0.02,
        "time_shift_type": "exponential",
        "use_beta_sigmas": False,
        "use_dynamic_shifting": True,
        "use_exponential_sigmas": False,
        "use_karras_sigmas": False,
    },
    "z-image": {
        "num_train_timesteps": 1000,
        "shift": 3.0,
        "use_dynamic_shifting": False,
    },
}

_AITK_SCHEDULER_BACKEND = None
_AITK_SCHEDULER_BACKEND_FAILED = False


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


def _apply_dit_flow_patch(model, model_type, shift, resolution_aware, width, height):
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
    return m


def _build_sigmas_formula(
    model_type,
    steps,
    width,
    height,
    shift,
    resolution_aware,
    force_aitk_timesteps,
    append_zero,
):
    config = AITK_FLOWMATCH_CONFIG[model_type]
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

    sigmas = sigmas.to(dtype=torch.float32, device="cpu")
    if append_zero:
        sigmas = torch.cat([sigmas, sigmas.new_zeros((1,))])
    return sigmas


def _load_aitk_scheduler_backend():
    global _AITK_SCHEDULER_BACKEND, _AITK_SCHEDULER_BACKEND_FAILED
    if _AITK_SCHEDULER_BACKEND is not None:
        return _AITK_SCHEDULER_BACKEND
    if _AITK_SCHEDULER_BACKEND_FAILED:
        return None

    try:
        repo_dir = os.path.dirname(os.path.abspath(__file__))
        candidates = (
            os.path.join(repo_dir, "ai-toolkit"),
            os.path.join(os.path.dirname(repo_dir), "ai-toolkit"),
        )
        for candidate in candidates:
            if os.path.isdir(candidate) and candidate not in sys.path:
                sys.path.insert(0, candidate)

        from toolkit.samplers.custom_flowmatch_sampler import (  # pylint: disable=import-outside-toplevel
            CustomFlowMatchEulerDiscreteScheduler,
            calculate_shift as aitk_calculate_shift,
        )
    except Exception:
        _AITK_SCHEDULER_BACKEND_FAILED = True
        return None

    _AITK_SCHEDULER_BACKEND = (CustomFlowMatchEulerDiscreteScheduler, aitk_calculate_shift)
    return _AITK_SCHEDULER_BACKEND


def _build_sigmas_exact(
    model_type,
    steps,
    width,
    height,
    shift,
    resolution_aware,
    force_aitk_timesteps,
    append_zero,
):
    backend = _load_aitk_scheduler_backend()
    if backend is None:
        return None

    try:
        import inspect  # pylint: disable=import-outside-toplevel
        import numpy as np  # pylint: disable=import-outside-toplevel

        scheduler_cls, aitk_calculate_shift = backend
        scheduler = scheduler_cls(**AITK_SCHEDULER_CONFIG[model_type])
        steps = max(int(steps), 1)
        params = set(inspect.signature(scheduler.set_timesteps).parameters)

        kwargs = {}
        if "device" in params:
            kwargs["device"] = "cpu"
        if "num_inference_steps" in params:
            kwargs["num_inference_steps"] = steps
        if "sigmas" in params and force_aitk_timesteps:
            kwargs["sigmas"] = np.linspace(1.0, 1.0 / float(steps), steps, dtype=np.float32)
        if "mu" in params and resolution_aware and AITK_SCHEDULER_CONFIG[model_type].get("use_dynamic_shifting", False):
            seq_len = (width // 16) * (height // 16)
            kwargs["mu"] = aitk_calculate_shift(
                seq_len,
                AITK_SCHEDULER_CONFIG[model_type].get("base_image_seq_len", 256),
                AITK_SCHEDULER_CONFIG[model_type].get("max_image_seq_len", 4096),
                AITK_SCHEDULER_CONFIG[model_type].get("base_shift", 0.5),
                AITK_SCHEDULER_CONFIG[model_type].get("max_shift", 1.15),
            )
        if "shift" in params and shift > 0.0 and not AITK_SCHEDULER_CONFIG[model_type].get("use_dynamic_shifting", False):
            kwargs["shift"] = float(shift)

        scheduler.set_timesteps(**kwargs)
        sigmas = scheduler.sigmas.detach().to(dtype=torch.float32, device="cpu")

        if append_zero:
            if sigmas.numel() == 0 or not torch.isclose(sigmas[-1], sigmas.new_zeros(()), atol=1e-6):
                sigmas = torch.cat([sigmas, sigmas.new_zeros((1,))])
        else:
            if sigmas.numel() > 0 and (
                torch.isclose(sigmas[-1], sigmas.new_zeros(()), atol=1e-6)
                or torch.isclose(sigmas[-1], sigmas.new_ones(()), atol=1e-6)
            ):
                sigmas = sigmas[:-1]
        return sigmas
    except Exception:
        return None


def _build_flowmatch_sigmas(
    model_type,
    steps,
    width,
    height,
    shift,
    resolution_aware,
    force_aitk_timesteps,
    append_zero,
):
    sigmas = _build_sigmas_exact(
        model_type,
        steps,
        width,
        height,
        shift,
        resolution_aware,
        force_aitk_timesteps,
        append_zero,
    )
    if sigmas is not None:
        return sigmas

    return _build_sigmas_formula(
        model_type,
        steps,
        width,
        height,
        shift,
        resolution_aware,
        force_aitk_timesteps,
        append_zero,
    )


class AIToolkitFlowMatchSampler:
    """
    All-in-one ai-toolkit style flowmatch sampler node.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "noise_seed": ("INT", {"default": 42, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "control_after_generate": True}),
                "add_noise": (["enable", "disable"], {"default": "enable"}),
                "steps": ("INT", {"default": 8, "min": 1, "max": 10000}),
                "guidance_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "sampler_name": (["euler", "res_multistep"], {"default": "euler"}),
                "model_type": (["auto", "flux", "z-image", "qwen"], {"default": "z-image"}),
                "width": ("INT", {"default": 768, "min": 64, "max": 8192, "step": 8}),
                "height": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 8}),
                "shift": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "resolution_aware": ("BOOLEAN", {"default": False}),
                "force_aitk_timesteps": ("BOOLEAN", {"default": True}),
                "append_zero": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("LATENT", "LATENT", "SIGMAS", "MODEL")
    RETURN_NAMES = ("output", "denoised_output", "sigmas", "patched_model")
    FUNCTION = "sample"
    CATEGORY = "sampling/custom_sampling"

    def sample(
        self,
        model,
        positive,
        negative,
        latent_image,
        noise_seed,
        add_noise,
        steps,
        guidance_scale,
        sampler_name,
        model_type,
        width,
        height,
        shift,
        resolution_aware,
        force_aitk_timesteps,
        append_zero,
    ):
        patched_model = _apply_dit_flow_patch(model, model_type, shift, resolution_aware, width, height)
        resolved_model_type = _resolve_model_type(patched_model, model_type)
        sigmas = _build_flowmatch_sigmas(
            resolved_model_type,
            steps,
            width,
            height,
            shift,
            resolution_aware,
            force_aitk_timesteps,
            append_zero,
        )

        latent = latent_image.copy()
        latent_samples = latent["samples"]
        latent_samples = comfy.sample.fix_empty_latent_channels(
            patched_model, latent_samples, latent.get("downscale_ratio_spacial", None)
        )
        latent["samples"] = latent_samples

        if add_noise == "disable":
            noise = torch.zeros(
                latent_samples.size(),
                dtype=latent_samples.dtype,
                layout=latent_samples.layout,
                device="cpu",
            )
        else:
            batch_inds = latent["batch_index"] if "batch_index" in latent else None
            noise = comfy.sample.prepare_noise(latent_samples, noise_seed, batch_inds)

        noise_mask = latent.get("noise_mask", None)

        guider = comfy.samplers.CFGGuider(patched_model)
        guider.set_conds(positive, negative)
        guider.set_cfg(guidance_scale)
        sampler = comfy.samplers.ksampler(sampler_name)

        x0_output = {}
        callback = latent_preview.prepare_callback(patched_model, max(sigmas.shape[-1] - 1, 0), x0_output)
        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED

        samples = guider.sample(
            noise,
            latent_samples,
            sampler,
            sigmas,
            denoise_mask=noise_mask,
            callback=callback,
            disable_pbar=disable_pbar,
            seed=noise_seed,
        )
        samples = samples.to(comfy.model_management.intermediate_device())

        out = latent.copy()
        out.pop("downscale_ratio_spacial", None)
        out["samples"] = samples

        if "x0" in x0_output:
            x0_out = guider.model_patcher.model.process_latent_out(x0_output["x0"].cpu())
            if samples.is_nested:
                latent_shapes = [x.shape for x in samples.unbind()]
                x0_out = comfy.nested_tensor.NestedTensor(comfy.utils.unpack_latents(x0_out, latent_shapes))
            out_denoised = latent.copy()
            out_denoised["samples"] = x0_out
        else:
            out_denoised = out

        return (out, out_denoised, sigmas, patched_model)


NODE_CLASS_MAPPINGS = {
    "AIToolkitFlowMatchSampler": AIToolkitFlowMatchSampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AIToolkitFlowMatchSampler": "FlowMatch Sampler (ai-toolkit exact)",
}

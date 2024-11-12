import torch
from typing import Tuple, Callable
import math
from typing import Type, Dict, Any, Tuple, Callable
import torch.nn.functional as F
from diffusers.models.attention import _chunked_feed_forward
import random
import time
from tabulate import tabulate
from merge import do_nothing, merge_wavg, merge_source, global_merge_1d, global_merge_2d, local_merge_2d, local_merge_2d_random
# from fvcore.nn import FlopCountAnalysis, parameter_count_table, flop_count_table
from diffusers.pipelines.stable_diffusion_3.pipeline_output import StableDiffusion3PipelineOutput
from typing import Any, Callable, Dict, List, Optional, Union
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import retrieve_timesteps
import os
from diffusers.utils import (
    USE_PEFT_BACKEND,
    is_torch_xla_available,
    logging,
    replace_example_docstring,
    scale_lora_layers,
    unscale_lora_layers,
)
if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False

global METRIC_SAVE
METRIC_SAVE = {}

# from counter import TIME_OTHERS, TIME_COMPUTE_MERGE, TIME_MERGE_ATTN, TIME_ATTN, TIME_UNMERGE_ATTN, TIME_MERGE_MLP, TIME_MLP, TIME_UNMERGE_MLP
from counter import TimeTracker
timetracker = TimeTracker.get_instance()
def isinstance_str(x: object, cls_name: str):
    """
    Checks whether x has any class *named* cls_name in its ancestry.
    Doesn't require access to the class's implementation.
    
    Useful for patching!
    """

    for _cls in x.__class__.__mro__:
        if _cls.__name__ == cls_name:
            return True
    
    return False

def init_generator(device: torch.device, fallback: torch.Generator=None):
    """
    Forks the current default random generator given device.
    """
    if device.type == "cpu":
        return torch.Generator(device="cpu").set_state(torch.get_rng_state())
    elif device.type == "cuda":
        return torch.Generator(device=device).set_state(torch.cuda.get_rng_state())
    else:
        if fallback is None:
            return init_generator(torch.device("cpu"))
        else:
            return fallback

def remove_patch(model: torch.nn.Module):
    """ Removes a patch from a ToMe Diffusion module if it was already patched. """
    # For diffusers
    model = model.unet if hasattr(model, "unet") else model.transformer if hasattr(model, "transformer") else model

    for _, module in model.named_modules():
        if hasattr(module, "_tome_info"):
            for hook in module._tome_info["hooks"]:
                hook.remove()
            module._tome_info["hooks"].clear()

        if module.__class__.__name__ == "ToMeBlock":
            module.__class__ = module._parent
    
    return model

def compute_ratio(tome_info):
    if "step_tmp_force" in tome_info:
        tome_info["step_tmp"] = tome_info["step_tmp_force"]
    step_tmp = tome_info["step_tmp"]
    layer_tmp = tome_info["layer_tmp"]
    if "unmerge_steps" in tome_info["args"] and step_tmp in tome_info["args"]["unmerge_steps"]:
        # print(step_tmp, layer_tmp)
        ratio_tmp = 0
    elif "unmerge_layers" in tome_info["args"] and layer_tmp in tome_info["args"]["unmerge_layers"]:
        # print(step_tmp, layer_tmp)
        ratio_tmp = 0
    else:
        ratio_start = tome_info["args"]["ratio_start"]
        ratio_end = tome_info["args"]["ratio_end"]
        step_count = tome_info["step_count"]

        ratio_tmp = 1 - (ratio_start + (ratio_end - ratio_start) / (step_count - 1) * step_tmp)

    return ratio_tmp

def compute_merge(x: torch.Tensor, tome_info: Dict[str, Any]) -> Tuple[Callable, ...]:
    args = tome_info["args"]

    w = int(math.sqrt(x.shape[1]))
    h = w
    assert w * h == x.shape[1], "Input must be square"
    tmp_ratio = compute_ratio(tome_info)
    # print(tmp_ratio)
    # print(x.shape)
    reduce_num = int(x.shape[1] * tmp_ratio)

    # Re-init the generator if it hasn't already been initialized or device has changed.
    if args["generator"] is None:
        args["generator"] = init_generator(x.device)
    elif args["generator"].device != x.device:
        args["generator"] = init_generator(x.device, fallback=args["generator"])
    
    # If the batch size is odd, then it's not possible for prompted and unprompted images to be in the same
    # batch, which causes artifacts with use_rand, so force it to be off.
    use_rand = False if x.shape[0] % 2 == 1 else args["use_rand"]

    # m, mp, u  = global_merge_1d(metric=x, layer_idx=tome_info["layer_tmp"], reduce_num=reduce_num, 
    #                             no_rand=not use_rand, generator=args["generator"], tome_info=tome_info)
    # m, mp, u  = local_merge_2d(metric=x, reduce_num=reduce_num, no_rand=not use_rand, generator=args["generator"], 
    #                            window_size=(args["sx"], args["sy"]), tome_info=tome_info)
    # m, mp, u  = local_merge_2d_random(metric=x, reduce_num=reduce_num, no_rand=not use_rand, generator=args["generator"], 
    #                                   window_size=(args["sx"], args["sy"]), tome_info=tome_info)
    # m, mp, u  = global_merge_2d(x, w, h, args["sx"], args["sy"], reduce_num=reduce_num, 
    #                                 no_rand=not use_rand, generator=args["generator"], tome_info=tome_info)
    if args["tome_type"] == "ToMe":
        m, mp, u  = global_merge_2d(x, w, h, args["sx"], args["sy"], reduce_num=reduce_num, 
                                no_rand=not use_rand, generator=args["generator"], tome_info=tome_info)
    elif args["tome_type"] == "ToMe_STD":
        if tome_info["step_tmp"] in tome_info["args"]["unmerge_steps"] or tome_info["args"]["metric_times"] == 1:
            if tome_info["step_tmp"] < args["STD_step"]:
                m, mp, u  = local_merge_2d(metric=x, reduce_num=reduce_num,window_size=(args["sx"], args["sy"]),  
                                        no_rand=not use_rand, generator=args["generator"], tome_info=tome_info)
            else:
                m, mp, u  = global_merge_2d(x, w, h, args["sx"], args["sy"], reduce_num=reduce_num, 
                                            no_rand=not use_rand, generator=args["generator"], tome_info=tome_info)
        elif tome_info["step_tmp"] % tome_info["args"]["metric_times"] == 0 or tome_info["layer_tmp"] not in tome_info["metric_save"]:
            if tome_info["step_tmp"] < args["STD_step"]:
                m, mp, u  = local_merge_2d(metric=x, reduce_num=reduce_num,window_size=(args["sx"], args["sy"]),  
                                        no_rand=not use_rand, generator=args["generator"], tome_info=tome_info)
            else:
                m, mp, u  = global_merge_2d(x, w, h, args["sx"], args["sy"], reduce_num=reduce_num, 
                                            no_rand=not use_rand, generator=args["generator"], tome_info=tome_info)
            if tome_info["args"]["mac_test"]:
                METRIC_SAVE[tome_info["layer_tmp"]] = (m, mp, u) 
            else:
                tome_info["metric_save"][tome_info["layer_tmp"]] = (m, mp, u)
        else:
            if tome_info["args"]["mac_test"]:
                (m, mp, u) = METRIC_SAVE[tome_info["layer_tmp"]]
            else:
                (m, mp, u) = tome_info["metric_save"][tome_info["layer_tmp"]]

    if args["prune_replace"] and args["step_tmp"] >= args["STD_step"]:
        m_a, u_a = (mp, u) if args["merge_attn"]      else (do_nothing, do_nothing)
        m_c, u_c = (mp, u) if args["merge_crossattn"] else (do_nothing, do_nothing)
        m_m, u_m = (mp, u) if args["merge_mlp"]       else (do_nothing, do_nothing)
    else:
        m_a, u_a = (m, u) if args["merge_attn"]      else (do_nothing, do_nothing)
        m_c, u_c = (m, u) if args["merge_crossattn"] else (do_nothing, do_nothing)
        m_m, u_m = (m, u) if args["merge_mlp"]       else (do_nothing, do_nothing)

    return m_a, m_c, m_m, u_a, u_c, u_m, mp  # Okay this is probably not very good

def make_tome_pipe(pipe_class: Type[torch.nn.Module]) -> Type[torch.nn.Module]:

    class FluxPipeline(pipe_class):
        # Save for unpatching later
        _parent = pipe_class
        # Global variables for timing
        def __call__(self, *args, **kwargs):
            # 更新自己及孩子的tome_info
            self._tome_info["step_count"] = kwargs['num_inference_steps']
            self._tome_info["step_iter"] = list(range(kwargs['num_inference_steps']))
            self.transformer._tome_info = self._tome_info
            output = super().__call__(*args, **kwargs)
            return output

    return FluxPipeline

def make_tome_fluxdit(model_class: Type[torch.nn.Module]) -> Type[torch.nn.Module]:
    
    class FluxTransformer2DModel_ToMe(model_class):
        # Save for unpatching later
        _parent = model_class
        # Global variables for timing
        def forward(self, *args, **kwargs):
            self._tome_info["layer_count"] = self.config.num_layers
            self._tome_info["step_tmp"] = self._tome_info["step_iter"].pop(0)
            self._tome_info["layer_iter"] = list(range(self.config.num_layers))
            output = super().forward(*args, **kwargs)
            return output

    return FluxTransformer2DModel_ToMe

def make_tome_block_fluxdit(block_class: Type[torch.nn.Module]) -> Type[torch.nn.Module]:
    """
    Make a patched class for a diffusers model.
    This patch applies ToMe to the forward function of the block.
    """
    class FluxTransformerBlock_ToMe(block_class):
        # Save for unpatching later
        _parent = block_class
        # Global variables for timing
        def forward(
            self, hidden_states: torch.FloatTensor, encoder_hidden_states: torch.FloatTensor, temb: torch.FloatTensor, image_rotary_emb=None,
        ):
            self._tome_info["layer_tmp"] = self._tome_info["layer_iter"].pop(0)
            step_tmp = self._tome_info["step_tmp"]
            layer_tmp = self._tome_info["layer_tmp"]
            # NOTE 1: Normalization 1
            start_time = time.time()
            norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, emb=temb)
            timetracker.log_time(start_time, "norm1", step_tmp, layer_tmp)

            # NOTE 2: Compute merge
            start_time = time.time()
            m_a, _, m_m, u_a, _, u_m, mp = compute_merge(norm_hidden_states, self._tome_info)
            timetracker.log_time(start_time, "compute_merge", step_tmp, layer_tmp)

            # NOTE 3: Normalization Context 1
            start_time = time.time()
            norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(
                encoder_hidden_states, emb=temb
            )
            timetracker.log_time(start_time, "norm1_context", step_tmp, layer_tmp)

            # NOTE 4: Merge Self-Attn
            start_time = time.time()
            if self._tome_info['args']['trace_source']:
                tmp_dict = {f"{self._tome_info['step_tmp']:02}" + '_' + f"{self._tome_info['layer_tmp']:02}" : merge_source(m_a, norm_hidden_states, None)}
                # print(tmp_dict)
                # self._tome_info['source'].append(tmp_dict)
                
            norm_hidden_states, _ = merge_wavg(m_a, norm_hidden_states)
            
            image_rotary_emb_tensor = torch.stack(image_rotary_emb, dim=0)
            txt_rotary_emb_tensor, img_rotary_emb_tensor = image_rotary_emb_tensor[:, :encoder_hidden_states.shape[1]], image_rotary_emb_tensor[:, encoder_hidden_states.shape[1]:]
            merge_img_rotary_emb = mp(img_rotary_emb_tensor)
            image_rotary_emb_tensor = torch.cat((txt_rotary_emb_tensor, merge_img_rotary_emb), dim=1)
            image_rotary_emb = image_rotary_emb_tensor.unbind(0)

            # for i in range(len(image_rotary_emb)):
            #     txt_rotary_emb, img_rotary_emb = image_rotary_emb[i][:encoder_hidden_states.shape[1]], image_rotary_emb[i][encoder_hidden_states.shape[1]:]
            #     merge_img_rotary_emb, _ = merge_wavg(m_a, img_rotary_emb.unsqueeze(0), None)
            #     merge_img_rotary_emb = merge_img_rotary_emb[0]
            #     image_rotary_emb[i] = torch.cat((txt_rotary_emb, merge_img_rotary_emb), dim=0)

            timetracker.log_time(start_time, "merge_attn", step_tmp, layer_tmp)

            # NOTE 5: Attention.    
            start_time = time.time()
            attn_output, context_attn_output = self.attn(
                hidden_states=norm_hidden_states,
                encoder_hidden_states=norm_encoder_hidden_states,
                image_rotary_emb=image_rotary_emb,
            )

            # Process attention outputs for the `hidden_states`.
            attn_output = gate_msa.unsqueeze(1) * attn_output
            timetracker.log_time(start_time, "attn", step_tmp, layer_tmp)

            # NOTE 6: Unmerge Self-Attn
            start_time = time.time()
            attn_output = u_a(attn_output)
            timetracker.log_time(start_time, "unmerge_attn", step_tmp, layer_tmp)
            
            hidden_states = hidden_states + attn_output

            # NOTE 7: Normalization 2
            start_time = time.time()
            norm_hidden_states = self.norm2(hidden_states)
            norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
            timetracker.log_time(start_time, "norm2", step_tmp, layer_tmp)

            # NOTE 8: Merge MLP
            start_time = time.time()
            norm_hidden_states, _ = merge_wavg(m_m, norm_hidden_states)
            # norm_hidden_states = m_m(norm_hidden_states)
            timetracker.log_time(start_time, "merge_mlp", step_tmp, layer_tmp)

            # NOTE 9: MLP
            start_time = time.time()
            ff_output = self.ff(norm_hidden_states)
            ff_output = gate_mlp.unsqueeze(1) * ff_output
            timetracker.log_time(start_time, "mlp", step_tmp, layer_tmp)

            # NOTE 10: Unmerge MLP
            start_time = time.time()
            ff_output = u_m(ff_output)
            timetracker.log_time(start_time, "unmerge_mlp", step_tmp, layer_tmp)

            hidden_states = hidden_states + ff_output

            # Process attention outputs for the `encoder_hidden_states`.
            # NOTE 11: Normalization Context 2
            start_time = time.time()
            context_attn_output = c_gate_msa.unsqueeze(1) * context_attn_output
            encoder_hidden_states = encoder_hidden_states + context_attn_output

            norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
            norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]

            context_ff_output = self.ff_context(norm_encoder_hidden_states)
            encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output
            if encoder_hidden_states.dtype == torch.float16:
                encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)
            timetracker.log_time(start_time, "norm2_context", step_tmp, layer_tmp)

            return encoder_hidden_states, hidden_states

    return FluxTransformerBlock_ToMe

def apply_patch_ToMe(
        pipe: torch.nn.Module,
        ratio: float = 0.5,
        sx: int = 2, sy: int = 2,
        use_rand: bool = True,
        merge_attn: bool = True,
        merge_crossattn: bool = False,
        merge_mlp: bool = False,
        merge_x: bool = False,
        ratio_start: float = 0,
        ratio_end: float = 0,
        save_dir: str = None,
        prune_replace: bool = False,
        trace_source: bool = False,
        STD_step: int = 0,
        speed_test: bool = False,
        mac_test: bool = False,
        unmerge_steps: list = [],
        unmerge_layers: list = [],
        tome_type: str = "ToMe",
        metric_times: int = 1,):
    """
    Patches a stable diffusion model with ToMe.
    Apply this to the highest level stable diffusion object (i.e., it should have a .model.diffusion_model).

    Important Args:
     - model: A top level Stable Diffusion module to patch in place. Should have a ".model.diffusion_model"
     - ratio: The ratio of tokens to merge. I.e., 0.4 would reduce the total number of tokens by 40%.
              The maximum value for this is 1-(1/(sx*sy)). By default, the max is 0.75 (I recommend <= 0.5 though).
              Higher values result in more speed-up, but with more visual quality loss.
    
    Args to tinker with if you want:
     - max_downsample [1, 2, 4, or 8]: Apply ToMe to layers with at most this amount of downsampling.
                                       E.g., 1 only applies to layers with no downsampling (4/15) while
                                       8 applies to all layers (15/15). I recommend a value of 1 or 2.
     - sx, sy: The stride for computing dst sets (see paper). A higher stride means you can merge more tokens,
               but the default of (2, 2) works well in most cases. Doesn't have to divide image size.
     - use_rand: Whether or not to allow random perturbations when computing dst sets (see paper). Usually
                 you'd want to leave this on, but if you're having weird artifacts try turning this off.
     - merge_attn: Whether or not to merge tokens for attention (recommended).
     - merge_crossattn: Whether or not to merge tokens for cross attention (not recommended).
     - merge_mlp: Whether or not to merge tokens for the mlp layers (very not recommended).
    """

    # Make sure the module is not currently patched
    remove_patch(pipe)
    make_pipe_fn = make_tome_pipe
    pipe.__class__ = make_pipe_fn(pipe.__class__)
    pipe._tome_info = {
        "step_count": None,
        "step_iter": None,
        "step_tmp": None,
        "layer_count": None,
        "layer_iter": None,
        "layer_tmp": None,
        "source": [],
        "metric_save": {},
        "args": {
            "save_dir" : save_dir,
            "ratio_start": ratio_start,
            "ratio_end": ratio_end,
            "ratio": ratio,
            "sx": sx, "sy": sy,
            "use_rand": use_rand,
            "generator": None,
            "tome_type": tome_type,
            "merge_attn": merge_attn,
            "merge_crossattn": merge_crossattn,
            "merge_mlp": merge_mlp,
            "merge_x": merge_x, 
            "prune_replace": prune_replace,
            "STD_step": STD_step,
            "trace_source": trace_source,
            "speed_test": speed_test,
            "mac_test": mac_test,
            "unmerge_steps": unmerge_steps,
            "unmerge_layers": unmerge_layers,
            "metric_times": metric_times,
        }
    }
    fluxdit = pipe.transformer
    make_fluxdit_fn = make_tome_fluxdit
    fluxdit.__class__ = make_fluxdit_fn(fluxdit.__class__)
    fluxdit._tome_info = pipe._tome_info
    for _, module in fluxdit.named_modules():
        if isinstance_str(module, "FluxTransformerBlock"):
            make_tome_block_fn = make_tome_block_fluxdit
            module.__class__ = make_tome_block_fn(module.__class__)
            module._tome_info = pipe._tome_info
    return pipe


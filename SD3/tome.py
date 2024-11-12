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

    return m_a, m_c, m_m, u_a, u_c, u_m  # Okay this is probably not very good

def make_tome_pipe(pipe_class: Type[torch.nn.Module]) -> Type[torch.nn.Module]:

    class StableDiffusion3Pipeline_ToMe(pipe_class):
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

    return StableDiffusion3Pipeline_ToMe

def make_tome_mmdit(model_class: Type[torch.nn.Module]) -> Type[torch.nn.Module]:
    
    class SD3Transformer2DModel_ToMe(model_class):
        # Save for unpatching later
        _parent = model_class
        # Global variables for timing
        def forward(self, *args, **kwargs):
            self._tome_info["layer_count"] = self.config.num_layers
            self._tome_info["step_tmp"] = self._tome_info["step_iter"].pop(0)
            self._tome_info["layer_iter"] = list(range(self.config.num_layers))
            output = super().forward(*args, **kwargs)
            return output

    return SD3Transformer2DModel_ToMe

def make_tome_block_mmdit(block_class: Type[torch.nn.Module]) -> Type[torch.nn.Module]:
    """
    Make a patched class for a diffusers model.
    This patch applies ToMe to the forward function of the block.
    """
    class JointTransformerBlock_ToMe(block_class):
        # Save for unpatching later
        _parent = block_class
        # Global variables for timing
        def forward(
            self, hidden_states: torch.FloatTensor, encoder_hidden_states: torch.FloatTensor, temb: torch.FloatTensor
        ):  
            self._tome_info["layer_tmp"] = self._tome_info["layer_iter"].pop(0)
            step_tmp = self._tome_info["step_tmp"]
            layer_tmp = self._tome_info["layer_tmp"]
            
            start_time = time.time()
            norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, emb=temb)
            timetracker.log_time(start_time, "norm1", step_tmp, layer_tmp)
            
            # (1) ToMe
            start_time = time.time()
            m_a, _, m_m, u_a, _, u_m = compute_merge(norm_hidden_states, self._tome_info)
            timetracker.log_time(start_time, "compute_merge", step_tmp, layer_tmp)

            start_time = time.time()
            if self.context_pre_only:
                norm_encoder_hidden_states = self.norm1_context(encoder_hidden_states, temb)
            else:
                norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(
                    encoder_hidden_states, emb=temb
                )
            timetracker.log_time(start_time, "norm1_context", step_tmp, layer_tmp)

            # (2) ToMe m_a
            start_time = time.time()
            if self._tome_info['args']['trace_source']:
                tmp_dict = {f"{self._tome_info['step_tmp']:02}" + '_' + f"{self._tome_info['layer_tmp']:02}" : merge_source(m_a, norm_hidden_states, None)}
                # print(tmp_dict)
                # self._tome_info['source'].append(tmp_dict)
                
            norm_hidden_states, _ = merge_wavg(m_a, norm_hidden_states)
            # norm_hidden_states = m_a(norm_hidden_states)
            timetracker.log_time(start_time, "merge_attn", step_tmp, layer_tmp)
            
            # Attention.
            start_time = time.time()
            attn_output, context_attn_output = self.attn(
                hidden_states=norm_hidden_states, encoder_hidden_states=norm_encoder_hidden_states
            )
            # Process attention outputs for the `hidden_states`.
            attn_output = gate_msa.unsqueeze(1) * attn_output
            timetracker.log_time(start_time, "attn", step_tmp, layer_tmp)

            # (3) ToMe u_a
            start_time = time.time()
            attn_output = u_a(attn_output)
            timetracker.log_time(start_time, "unmerge_attn", step_tmp, layer_tmp)

            hidden_states = hidden_states + attn_output

            start_time = time.time()
            norm_hidden_states = self.norm2(hidden_states)
            norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
            timetracker.log_time(start_time, "norm2", step_tmp, layer_tmp)
            
            
            # (4) ToMe m_m
            start_time = time.time()
            norm_hidden_states, _ = merge_wavg(m_m, norm_hidden_states)
            # norm_hidden_states = m_m(norm_hidden_states)
            timetracker.log_time(start_time, "merge_mlp", step_tmp, layer_tmp)

            start_time = time.time()
            if self._chunk_size is not None:
                # "feed_forward_chunk_size" can be used to save memory
                ff_output = _chunked_feed_forward(self.ff, norm_hidden_states, self._chunk_dim, self._chunk_size)
            else:
                ff_output = self.ff(norm_hidden_states)
            ff_output = gate_mlp.unsqueeze(1) * ff_output
            timetracker.log_time(start_time, "mlp", step_tmp, layer_tmp)

            # (5) ToMe u_m
            start_time = time.time()
            ff_output = u_m(ff_output)
            timetracker.log_time(start_time, "unmerge_mlp", step_tmp, layer_tmp)

            hidden_states = hidden_states + ff_output

            # Process attention outputs for the `encoder_hidden_states`.
            start_time = time.time()
            if self.context_pre_only:
                encoder_hidden_states = None
            else:
                context_attn_output = c_gate_msa.unsqueeze(1) * context_attn_output
                encoder_hidden_states = encoder_hidden_states + context_attn_output

                norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
                norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]
                if self._chunk_size is not None:
                    # "feed_forward_chunk_size" can be used to save memory
                    context_ff_output = _chunked_feed_forward(
                        self.ff_context, norm_encoder_hidden_states, self._chunk_dim, self._chunk_size
                    )
                else:
                    context_ff_output = self.ff_context(norm_encoder_hidden_states)
                encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output
            timetracker.log_time(start_time, "norm2_context", step_tmp, layer_tmp)
            
            return encoder_hidden_states, hidden_states

    return JointTransformerBlock_ToMe

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
    mmdit = pipe.transformer
    make_mmdit_fn = make_tome_mmdit
    mmdit.__class__ = make_mmdit_fn(mmdit.__class__)
    mmdit._tome_info = pipe._tome_info
    for _, module in mmdit.named_modules():
        if isinstance_str(module, "JointTransformerBlock"):
            make_tome_block_fn = make_tome_block_mmdit
            module.__class__ = make_tome_block_fn(module.__class__)
            module._tome_info = pipe._tome_info
    return pipe


# def make_tome_pipe(pipe_class: Type[torch.nn.Module]) -> Type[torch.nn.Module]:
#     # change forward to save stepimage and compute flops
#     class StableDiffusion3Pipeline_ToMe(pipe_class):
#         # Save for unpatching later
#         _parent = pipe_class
#         # Global variables for timing

#         def __call__(self,
#         prompt: Union[str, List[str]] = None,
#         prompt_2: Optional[Union[str, List[str]]] = None,
#         prompt_3: Optional[Union[str, List[str]]] = None,
#         height: Optional[int] = None,
#         width: Optional[int] = None,
#         num_inference_steps: int = 28,
#         timesteps: List[int] = None,
#         guidance_scale: float = 7.0,
#         negative_prompt: Optional[Union[str, List[str]]] = None,
#         negative_prompt_2: Optional[Union[str, List[str]]] = None,
#         negative_prompt_3: Optional[Union[str, List[str]]] = None,
#         num_images_per_prompt: Optional[int] = 1,
#         generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
#         latents: Optional[torch.FloatTensor] = None,
#         prompt_embeds: Optional[torch.FloatTensor] = None,
#         negative_prompt_embeds: Optional[torch.FloatTensor] = None,
#         pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
#         negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
#         output_type: Optional[str] = "pil",
#         return_dict: bool = True,
#         joint_attention_kwargs: Optional[Dict[str, Any]] = None,
#         clip_skip: Optional[int] = None,
#         callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
#         callback_on_step_end_tensor_inputs: List[str] = ["latents"],
#         max_sequence_length: int = 256,
#         save_step: bool = False,
#         speed_test: bool = False):

#             # 更新自己及孩子的tome_info
#             self._tome_info["step_count"] = num_inference_steps
#             self._tome_info["step_iter"] = list(range(num_inference_steps))
#             self.transformer._tome_info = self._tome_info

#             height = height or self.default_sample_size * self.vae_scale_factor
#             width = width or self.default_sample_size * self.vae_scale_factor

#             # 1. Check inputs. Raise error if not correct
#             self.check_inputs(
#                 prompt,
#                 prompt_2,
#                 prompt_3,
#                 height,
#                 width,
#                 negative_prompt=negative_prompt,
#                 negative_prompt_2=negative_prompt_2,
#                 negative_prompt_3=negative_prompt_3,
#                 prompt_embeds=prompt_embeds,
#                 negative_prompt_embeds=negative_prompt_embeds,
#                 pooled_prompt_embeds=pooled_prompt_embeds,
#                 negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
#                 callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
#                 max_sequence_length=max_sequence_length,
#             )

#             self._guidance_scale = guidance_scale
#             self._clip_skip = clip_skip
#             self._joint_attention_kwargs = joint_attention_kwargs
#             self._interrupt = False

#             # 2. Define call parameters
#             if prompt is not None and isinstance(prompt, str):
#                 batch_size = 1
#             elif prompt is not None and isinstance(prompt, list):
#                 batch_size = len(prompt)
#             else:
#                 batch_size = prompt_embeds.shape[0]

#             device = self._execution_device

#             lora_scale = (
#                 self.joint_attention_kwargs.get("scale", None) if self.joint_attention_kwargs is not None else None
#             )
#             (
#                 prompt_embeds,
#                 negative_prompt_embeds,
#                 pooled_prompt_embeds,
#                 negative_pooled_prompt_embeds,
#             ) = self.encode_prompt(
#                 prompt=prompt,
#                 prompt_2=prompt_2,
#                 prompt_3=prompt_3,
#                 negative_prompt=negative_prompt,
#                 negative_prompt_2=negative_prompt_2,
#                 negative_prompt_3=negative_prompt_3,
#                 do_classifier_free_guidance=self.do_classifier_free_guidance,
#                 prompt_embeds=prompt_embeds,
#                 negative_prompt_embeds=negative_prompt_embeds,
#                 pooled_prompt_embeds=pooled_prompt_embeds,
#                 negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
#                 device=device,
#                 clip_skip=self.clip_skip,
#                 num_images_per_prompt=num_images_per_prompt,
#                 max_sequence_length=max_sequence_length,
#                 lora_scale=lora_scale,
#             )

#             if self.do_classifier_free_guidance:
#                 prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
#                 pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)

#             # 4. Prepare timesteps
#             timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps)
#             num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
#             self._num_timesteps = len(timesteps)

#             # 5. Prepare latent variables
#             num_channels_latents = self.transformer.config.in_channels
#             latents = self.prepare_latents(
#                 batch_size * num_images_per_prompt,
#                 num_channels_latents,
#                 height,
#                 width,
#                 prompt_embeds.dtype,
#                 device,
#                 generator,
#                 latents,
#             )

#             # 6. Denoising loop
#             import time

#             if save_step:
#                 save_dir = self.transformer._tome_info["save_dir"]
#                 os.makedirs(save_dir, exist_ok=True)
#             # Start the timer manually
#             each_time = []

#             with self.progress_bar(total=num_inference_steps) as progress_bar:
#                 for i, t in enumerate(timesteps):
#                     if self.interrupt:
#                         continue

#                     # expand the latents if we are doing classifier free guidance
#                     latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
#                     # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
#                     timestep = t.expand(latent_model_input.shape[0])

#                     start_time = time.time()
#                     if speed_test:
#                         pass
#                         #NOTE: fvcore
#                         # from fvcore.nn import FlopCountAnalysis, flop_count_table
#                         # # 实例化包装模块
#                         # transformer_wrapper = TransformerWrapper(self.transformer)
#                         # inputs = (
#                         #     latent_model_input,
#                         #     timestep,
#                         #     prompt_embeds,
#                         #     pooled_prompt_embeds,
#                         #     self.joint_attention_kwargs,
#                         #     return_dict,
#                         # )
#                         # flops = FlopCountAnalysis(transformer_wrapper, inputs)
#                         # # flops = register_custom_flop_counters(flops)
#                         # # 打印总的 GFLOPs
#                         # print(f"Total FLOPs: {flops.total() / 1e9:.2f} GFLOPs")
#                         # # 打印每个模块的 FLOPs 细节
#                         # print(flop_count_table(flops))

#                         #NOTE: ptflops
#                         # from ptflops import get_model_complexity_info
#                         # model_wrapper = TransformerDynamicWrapper(self.transformer, timestep, prompt_embeds, pooled_prompt_embeds, self.joint_attention_kwargs)
                        
#                         # # Calculate FLOPs using ptflops
#                         # macs, params = get_model_complexity_info(model_wrapper, (latent_model_input.size(),), as_strings=False,
#                         #                                         print_per_layer_stat=True, verbose=True)

#                         # print(f"Model MACs (Multiply-Accumulate Operations): {macs}")
#                         # print(f"Model Parameters: {params}")

#                         #NOTE: deepcache
#                         # from flops import count_ops_and_params
#                         # example_inputs = {
#                         #     'hidden_states': latent_model_input, 
#                         #     'timestep': timestep,
#                         #     'encoder_hidden_states': prompt_embeds,
#                         #     'pooled_projections': pooled_prompt_embeds,
#                         #     'joint_attention_kwargs': self.joint_attention_kwargs,
#                         #     'return_dict': False,
#                         # }
#                         # macs, nparams = count_ops_and_params(self.transformer, example_inputs=example_inputs, layer_wise=False)
#                         # print("#Params: {:.4f} M".format(nparams/1e6))
#                         # print("#MACs: {:.4f} G".format(macs/1e9))

#                         #NOTE: thop
#                         # from thop import profile, clever_format
#     #                         def forward(
#                         #     self,
#                         #     hidden_states: torch.FloatTensor,
#                         #     encoder_hidden_states: torch.FloatTensor = None,
#                         #     pooled_projections: torch.FloatTensor = None,
#                         #     timestep: torch.LongTensor = None,
#                         #     block_controlnet_hidden_states: List = None,
#                         #     joint_attention_kwargs: Optional[Dict[str, Any]] = None,
#                         #     return_dict: bool = True,
#                         # ) -> Union[torch.FloatTensor, Transformer2DModelOutput]:
#                         # transformer_wrapper = TransformerWrapper(self.transformer)
#                         # inputs = (
#                         #     latent_model_input,
#                         #     prompt_embeds,
#                         #     pooled_prompt_embeds,
#                         #     timestep,
#                         #     None,
#                         #     self.joint_attention_kwargs,
#                         #     False,
#                         # )
#                         # flops, params = profile(self.transformer, inputs=inputs, verbose=True)

#                         # # 转换为GFLOPs
#                         # gflops = flops / 1e9
#                         # print(f"GFLOPS: {gflops:.2f} G")
#                         # print(f"Total Parameters: {params}")
#                         # # 代码执行完毕，清理变量
#                         # del inputs, flops, params, gflops
#                         # torch.cuda.empty_cache()  # 如果使用 GPU
                        
#                     noise_pred = self.transformer(
#                         hidden_states=latent_model_input,
#                         timestep=timestep,
#                         encoder_hidden_states=prompt_embeds,
#                         pooled_projections=pooled_prompt_embeds,
#                         joint_attention_kwargs=self.joint_attention_kwargs,
#                         return_dict=False,
#                     )[0]
                    
#                     each_time.append(time.time() - start_time)

#                     # perform guidance
#                     if self.do_classifier_free_guidance:
#                         noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
#                         noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

#                     # compute the previous noisy sample x_t -> x_t-1
#                     latents_dtype = latents.dtype
#                     latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

#                     if latents.dtype != latents_dtype:
#                         if torch.backends.mps.is_available():
#                             # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
#                             latents = latents.to(latents_dtype)

#                     if callback_on_step_end is not None:
#                         callback_kwargs = {}
#                         for k in callback_on_step_end_tensor_inputs:
#                             callback_kwargs[k] = locals()[k]
#                         callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

#                         latents = callback_outputs.pop("latents", latents)
#                         prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
#                         negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)
#                         negative_pooled_prompt_embeds = callback_outputs.pop(
#                             "negative_pooled_prompt_embeds", negative_pooled_prompt_embeds
#                         )

#                     # call the callback, if provided
#                     if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
#                         progress_bar.update()

#                     if XLA_AVAILABLE:
#                         xm.mark_step()

#                     # #! save every step
#                     # if save_step:
#                     #     latents_save = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor

#                     #     image_save = self.vae.decode(latents_save, return_dict=False)[0]
#                     #     image_save = self.image_processor.postprocess(image_save, output_type=output_type)
#                     #     image_save[0].save(os.path.join(save_dir, f"{i}.png"))
#             if speed_test:
#                 # for i, iter_time in enumerate(each_time):
#                 #     print(f"{i} spent: {iter_time:.3f} seconds")
#                 avg_time = sum(each_time) / len(each_time)
#                 print(f"Avg time spent: {avg_time} seconds")
#                 total_time = sum(each_time)
#                 print(f"Total time spent: {total_time} seconds")

#             if output_type == "latent":
#                 image = latents

#             else:
#                 latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor

#                 image = self.vae.decode(latents, return_dict=False)[0]
#                 image = self.image_processor.postprocess(image, output_type=output_type)

#             # Offload all models
#             self.maybe_free_model_hooks()

#             if not return_dict:
#                 return (image,)

#             return StableDiffusion3PipelineOutput(images=image)

#     return StableDiffusion3Pipeline_ToMe

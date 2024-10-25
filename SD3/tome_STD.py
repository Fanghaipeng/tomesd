import torch
from typing import Tuple, Callable
import math
from typing import Type, Dict, Any, Tuple, Callable
import torch.nn.functional as F
from diffusers.models.attention import _chunked_feed_forward
import random
import time
from tabulate import tabulate
from merge import do_nothing, merge_wavg, merge_source, global_merge_2d

TIME_OTHERS = 0
TIME_COMPUTE_MERGE = 0
TIME_MERGE_ATTN = 0
TIME_ATTN = 0
TIME_UNMERGE_ATTN = 0
TIME_MERGE_MLP = 0
TIME_MLP = 0
TIME_UNMERGE_MLP = 0

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
    ratio_start = tome_info["args"]["ratio_start"]
    ratio_end = tome_info["args"]["ratio_end"]
    step_count = tome_info["step_count"]
    step_tmp = tome_info["step_tmp"]
    ratio_tmp = 1 - (ratio_start + (ratio_end - ratio_start) / (step_count - 1) * step_tmp) # ratio 是裁剪ratio，1 - 好理解
    return ratio_tmp

def compute_merge(x: torch.Tensor, tome_info: Dict[str, Any]) -> Tuple[Callable, ...]:
    args = tome_info["args"]

    w = int(math.sqrt(x.shape[1]))
    h = w
    assert w * h == x.shape[1], "Input must be square"
    tmp_ratio = compute_ratio(tome_info)
    # print(tmp_ratio)
    # print(x.shape)
    r = int(x.shape[1] * tmp_ratio)

    # Re-init the generator if it hasn't already been initialized or device has changed.
    if args["generator"] is None:
        args["generator"] = init_generator(x.device)
    elif args["generator"].device != x.device:
        args["generator"] = init_generator(x.device, fallback=args["generator"])
    
    # If the batch size is odd, then it's not possible for prompted and unprompted images to be in the same
    # batch, which causes artifacts with use_rand, so force it to be off.
    use_rand = False if x.shape[0] % 2 == 1 else args["use_rand"]
    m, mp, u  = global_merge_2d(x, w, h, args["sx"], args["sy"], r, 
                                                    no_rand=not use_rand, generator=args["generator"])

    if args["prune_replace"] and args["step"] >= args["replace_step"]:
        m_a, u_a = (mp, u) if args["merge_attn"]      else (do_nothing, do_nothing)
        m_c, u_c = (mp, u) if args["merge_crossattn"] else (do_nothing, do_nothing)
        m_m, u_m = (mp, u) if args["merge_mlp"]       else (do_nothing, do_nothing)
    else:
        m_a, u_a = (m, u) if args["merge_attn"]      else (do_nothing, do_nothing)
        m_c, u_c = (m, u) if args["merge_crossattn"] else (do_nothing, do_nothing)
        m_m, u_m = (m, u) if args["merge_mlp"]       else (do_nothing, do_nothing)

    return m_a, m_c, m_m, u_a, u_c, u_m  # Okay this is probably not very good

def show_timing_info_STD():
    # 创建一个列表，包含所有时间数据和相应的标签
    data = [
        ["Norm and Res", TIME_OTHERS],
        ["Compute Merge", TIME_COMPUTE_MERGE],
        ["Merge Attention", TIME_MERGE_ATTN],
        ["Attention", TIME_ATTN],
        ["Unmerge Attention", TIME_UNMERGE_ATTN],
        ["Merge MLP", TIME_MERGE_MLP],
        ["MLP", TIME_MLP],
        ["Unmerge MLP", TIME_UNMERGE_MLP]
    ]

    # 使用 tabulate 打印表格，选择 "grid" 格式使表格更易读
    print(tabulate(data, headers=['Operation', 'Time (seconds)'], tablefmt='grid'))

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
            return super().__call__(*args, **kwargs)

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

            return super().forward(*args, **kwargs)

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
            # print(self._tome_info["step_tmp"], self._tome_info["layer_tmp"])
            global TIME_OTHERS, TIME_COMPUTE_MERGE, TIME_MERGE_ATTN, TIME_ATTN, TIME_UNMERGE_ATTN, \
                    TIME_MERGE_MLP, TIME_MLP, TIME_UNMERGE_MLP
            
            start_time = time.time()
            norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, emb=temb)
            TIME_OTHERS += time.time() - start_time
            
            # (1) ToMe
            start_time = time.time()
            m_a, _, m_m, u_a, _, u_m = compute_merge(norm_hidden_states, self._tome_info)
            TIME_COMPUTE_MERGE += time.time() - start_time

            start_time = time.time()
            if self.context_pre_only:
                norm_encoder_hidden_states = self.norm1_context(encoder_hidden_states, temb)
            else:
                norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(
                    encoder_hidden_states, emb=temb
                )
            TIME_OTHERS += time.time() - start_time

            # (2) ToMe m_a
            start_time = time.time()
            if self._tome_info['args']['trace_source']:
                tmp_dict = {f"{self._tome_info['step_tmp']:02}" + '_' + f"{self._tome_info['layer_tmp']:02}" : merge_source(m_a, norm_hidden_states, None)}
                print(tmp_dict)
                self._tome_info['source'].append(tmp_dict)
            norm_hidden_states = m_a(norm_hidden_states)

            TIME_MERGE_ATTN += time.time() - start_time
            
            # Attention.
            start_time = time.time()
            attn_output, context_attn_output = self.attn(
                hidden_states=norm_hidden_states, encoder_hidden_states=norm_encoder_hidden_states
            )
            # Process attention outputs for the `hidden_states`.
            attn_output = gate_msa.unsqueeze(1) * attn_output
            TIME_ATTN += time.time() - start_time

            # (3) ToMe u_a
            start_time = time.time()
            attn_output = u_a(attn_output)
            TIME_UNMERGE_ATTN += time.time() - start_time
            hidden_states = hidden_states + attn_output

            start_time = time.time()
            norm_hidden_states = self.norm2(hidden_states)
            TIME_OTHERS += time.time() - start_time

            # (4) ToMe m_m
            start_time = time.time()
            norm_hidden_states = m_m(norm_hidden_states)
            TIME_MERGE_MLP += time.time() - start_time

            start_time = time.time()
            norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
            TIME_OTHERS += time.time() - start_time

            start_time = time.time()
            if self._chunk_size is not None:
                # "feed_forward_chunk_size" can be used to save memory
                ff_output = _chunked_feed_forward(self.ff, norm_hidden_states, self._chunk_dim, self._chunk_size)
            else:
                ff_output = self.ff(norm_hidden_states)
            ff_output = gate_mlp.unsqueeze(1) * ff_output
            TIME_MLP += time.time() - start_time

            # (5) ToMe u_m
            start_time = time.time()
            ff_output = u_m(ff_output)
            TIME_UNMERGE_MLP += time.time() - start_time

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
            TIME_OTHERS += time.time() - start_time
            
            return encoder_hidden_states, hidden_states

    return JointTransformerBlock_ToMe

def apply_patch_ToMe_STD(
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
        replace_step: int = 0):
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
        "args": {
            "save_dir" : save_dir,
            "ratio_start": ratio_start,
            "ratio_end": ratio_end,
            "ratio": ratio,
            "sx": sx, "sy": sy,
            "use_rand": use_rand,
            "generator": None,
            "merge_attn": merge_attn,
            "merge_crossattn": merge_crossattn,
            "merge_mlp": merge_mlp,
            "merge_x": merge_x, 
            "prune_replace": prune_replace,
            "replace_step": replace_step,
            "trace_source": trace_source,
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



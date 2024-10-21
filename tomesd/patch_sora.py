import torch
import math
from typing import Type, Dict, Any, Tuple, Callable

from . import merge_sora as merge
from .utils import isinstance_str, init_generator
from torch import nn
from opensora.models.layers.blocks import t2i_modulate
from einops import rearrange

def compute_merge_spatial(x: torch.Tensor, tome_info: Dict[str, Any]) -> Tuple[Callable, ...]:
    original_t, original_s, original_h, original_w = tome_info["size"]
    original_tokens = original_h * original_w
    x = rearrange(x, "B (T S) C -> (B T) S C", T=original_t, S=original_s)
    downsample = int(math.ceil(math.sqrt(original_tokens // x.shape[1])))

    args = tome_info["args"]

    if downsample <= args["max_downsample"]:
        w = int(math.ceil(original_w / downsample))
        h = int(math.ceil(original_h / downsample))

        # Re-init the generator if it hasn't already been initialized or device has changed.
        if args["generator"] is None:
            args["generator"] = init_generator(x.device)
        elif args["generator"].device != x.device:
            args["generator"] = init_generator(x.device, fallback=args["generator"])
        
        # If the batch size is odd, then it's not possible for prompted and unprompted images to be in the same
        # batch, which causes artifacts with use_rand, so force it to be off.
        use_rand = False if x.shape[0] % 2 == 1 else args["use_rand"]

        m, u = merge.bipartite_soft_matching_random2d(x, w, h, args["sx"], args["sy"], args["ratio"], 
                                                      no_rand=not use_rand, generator=args["generator"])
    else:
        m, u = (merge.do_nothing, merge.do_nothing)

    m_a, u_a = (m, u) if args["merge_attn"]      else (merge.do_nothing, merge.do_nothing)
    m_c, u_c = (m, u) if args["merge_crossattn"] else (merge.do_nothing, merge.do_nothing)
    m_m, u_m = (m, u) if args["merge_mlp"]       else (merge.do_nothing, merge.do_nothing)

    return m_a, m_c, m_m, u_a, u_c, u_m  # Okay this is probably not very good

def compute_merge_temporal(x: torch.Tensor, tome_info: Dict[str, Any]) -> Tuple[Callable, ...]:
    original_t, original_s, original_h, original_w = tome_info["size"]
    original_tokens = original_t
    x = rearrange(x, "B (T S) C -> (B S) T C", T=original_t, S=original_s)
    downsample = int(math.ceil(math.sqrt(original_tokens // x.shape[1])))

    args = tome_info["args"]

    if downsample <= args["max_downsample"]:
        w = int(math.ceil(original_t / downsample))
        h = int(math.ceil(1 / downsample))

        # Re-init the generator if it hasn't already been initialized or device has changed.
        if args["generator"] is None:
            args["generator"] = init_generator(x.device)
        elif args["generator"].device != x.device:
            args["generator"] = init_generator(x.device, fallback=args["generator"])
        
        # If the batch size is odd, then it's not possible for prompted and unprompted images to be in the same
        # batch, which causes artifacts with use_rand, so force it to be off.
        use_rand = False if x.shape[0] % 2 == 1 else args["use_rand"]

        m, u = merge.bipartite_soft_matching_random2d(x, w, h, args["st"], 1, args["ratio"], 
                                                      no_rand=not use_rand, generator=args["generator"])
    else:
        m, u = (merge.do_nothing, merge.do_nothing)

    m_a, u_a = (m, u) if args["merge_attn"]      else (merge.do_nothing, merge.do_nothing)
    m_c, u_c = (m, u) if args["merge_crossattn"] else (merge.do_nothing, merge.do_nothing)
    m_m, u_m = (m, u) if args["merge_mlp"]       else (merge.do_nothing, merge.do_nothing)

    return m_a, m_c, m_m, u_a, u_c, u_m  # Okay this is probably not very good

def make_opensora_tome_block(block_class: Type[torch.nn.Module]) -> Type[torch.nn.Module]:
    """
    Make a patched class for a diffusers model.
    This patch applies ToMe to the forward function of the block.
    """
    class OpenSora_ToMeBlock(block_class):
        # Save for unpatching later
        _parent = block_class

        def forward(
            self,
            x,
            y,
            t,
            mask=None,  # text mask
            x_mask=None,  # temporal mask
            t0=None,  # t with timestamp=0
            T=None,  # number of frames
            S=None,  # number of pixel patches
            H=None,  # height of image
            W=None,  # width of image
        ):
            # prepare modulate parameters
            B, N, C = x.shape

            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                self.scale_shift_table[None] + t.reshape(B, 6, -1)
            ).chunk(6, dim=1)
            if x_mask is not None:
                shift_msa_zero, scale_msa_zero, gate_msa_zero, shift_mlp_zero, scale_mlp_zero, gate_mlp_zero = (
                    self.scale_shift_table[None] + t0.reshape(B, 6, -1)
                ).chunk(6, dim=1)

            # modulate (attention)
            x_m = t2i_modulate(self.norm1(x), shift_msa, scale_msa)
            if x_mask is not None:
                x_m_zero = t2i_modulate(self.norm1(x), shift_msa_zero, scale_msa_zero)
                x_m = self.t_mask_select(x_mask, x_m, x_m_zero, T, S)

            self._tome_info["size"] = (T, S, H, W)
            # (1) ToMe
            if self.temporal:
                m_a, m_c, m_m, u_a, u_c, u_m = compute_merge_temporal(x_m, self._tome_info)
            else:
                m_a, m_c, m_m, u_a, u_c, u_m = compute_merge_spatial(x_m, self._tome_info)

            # attention
            if self.temporal:
                x_m = rearrange(x_m, "B (T S) C -> (B S) T C", T=T, S=S)
                # (2) ToMe m_a
                x_m = m_a(x_m)

                x_m = self.attn(x_m)

                # (3) ToMe u_a
                x_m = u_a(x_m)

                x_m = rearrange(x_m, "(B S) T C -> B (T S) C", T=T, S=S)
            else:
                x_m = rearrange(x_m, "B (T S) C -> (B T) S C", T=T, S=S)
                # (2) ToMe m_a
                x_m = m_a(x_m)

                x_m = self.attn(x_m)

                # (3) ToMe u_a
                x_m = u_a(x_m)

                x_m = rearrange(x_m, "(B T) S C -> B (T S) C", T=T, S=S)
            
            # modulate (attention)
            x_m_s = gate_msa * x_m
            if x_mask is not None:
                x_m_s_zero = gate_msa_zero * x_m
                x_m_s = self.t_mask_select(x_mask, x_m_s, x_m_s_zero, T, S)

            # residual
            x = x + self.drop_path(x_m_s)

            # (4) ToMe m_c, u_c
            # cross attention
            x = x + u_c(self.cross_attn(m_c(x), y, mask))

            # modulate (MLP)
            x_m = t2i_modulate(self.norm2(x), shift_mlp, scale_mlp)
            if x_mask is not None:
                x_m_zero = t2i_modulate(self.norm2(x), shift_mlp_zero, scale_mlp_zero)
                x_m = self.t_mask_select(x_mask, x_m, x_m_zero, T, S)

            # (5) ToMe m_m
            x_m = m_m(x_m)
            # MLP
            x_m = self.mlp(x_m)
            # (6) ToMe u_m
            x_m = u_m(x_m)

            # modulate (MLP)
            x_m_s = gate_mlp * x_m
            if x_mask is not None:
                x_m_s_zero = gate_mlp_zero * x_m
                x_m_s = self.t_mask_select(x_mask, x_m_s, x_m_s_zero, T, S)

            # residual
            x = x + self.drop_path(x_m_s)

            return x
    

    return OpenSora_ToMeBlock

def apply_patch_opensora(
        model: torch.nn.Module,
        ratio: float = 0.5,
        spatial_start_ratio: float = 0.5,
        spatial_end_ratio: float = 0.5,
        temporal_start_ratio: float = 0.5,
        temporal_end_ratio: float = 0.5,
        max_downsample: int = 1,
        sx: int = 2, sy: int = 2, st: int = 4,
        use_rand: bool = True,
        merge_attn: bool = True,
        merge_crossattn: bool = False,
        merge_mlp: bool = False,):
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
    remove_patch_opensora(model)
    diffusion_model = model

    diffusion_model._tome_info = {
        "size": None,
        "hooks": [],
        "size": (),
        "args": {
            "ratio": ratio,
            "spatial_start_ratio": spatial_start_ratio,
            "spatial_end_ratio": spatial_end_ratio,
            "temporal_start_ratio": temporal_start_ratio,
            "temporal_end_ratio": temporal_end_ratio,
            "max_downsample": max_downsample,
            "sx": sx, "sy": sy, "st": st,
            "use_rand": use_rand,
            "generator": None,
            "merge_attn": merge_attn,
            "merge_crossattn": merge_crossattn,
            "merge_mlp": merge_mlp
        }
    }
    # hook_tome_model(diffusion_model)

    for _, module in diffusion_model.named_modules():
        if isinstance_str(module, "STDiT3Block"):
            make_tome_block_fn = make_opensora_tome_block 
            module.__class__ = make_tome_block_fn(module.__class__)
            module._tome_info = diffusion_model._tome_info

    return model


# def hook_tome_model(model: torch.nn.Module):
#     """ Adds a forward pre hook to get the image size. This hook can be removed with remove_patch. """
#     def hook(module, args):
#         if isinstance_str(module, "SD3Transformer2DModel") or isinstance_str(module, "SD3Transformer2DModel") or:
#             donothing = True
#         else:
#             module._tome_info["size"] = (args[0].shape[2], args[0].shape[3])
#         return None

#     model._tome_info["hooks"].append(model.register_forward_pre_hook(hook))

def remove_patch_opensora(model: torch.nn.Module):
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

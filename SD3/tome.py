import torch
from typing import Tuple, Callable
import math
from typing import Type, Dict, Any, Tuple, Callable
import torch.nn.functional as F
from diffusers.models.attention import _chunked_feed_forward
import random
import time
from tabulate import tabulate

time_norm1 = 0
time_compute_merge = 0
time_contextnorm = 0
time_merge_attn = 0
time_attn = 0
time_unmerge_attn = 0
time_norm2 = 0
time_merge_mlp = 0
time_mlp = 0
time_unmerge_mlp = 0

def do_nothing(x: torch.Tensor, mode:str=None):
    return x

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

def mps_gather_workaround(input, dim, index):
    if input.shape[-1] == 1:
        return torch.gather(
            input.unsqueeze(-1),
            dim - 1 if dim < 0 else dim,
            index.unsqueeze(-1)
        ).squeeze(-1)
    else:
        return torch.gather(input, dim, index)


def bipartite_soft_matching_random2d(metric: torch.Tensor,
                                     w: int, h: int, sx: int, sy: int, r: int,
                                     no_rand: bool = False,
                                     generator: torch.Generator = None) -> Tuple[Callable, Callable]:
    """
    Partitions the tokens into src and dst and merges r tokens from src to dst.
    Dst tokens are partitioned by choosing one randomy in each (sx, sy) region.

    Args:
     - metric [B, N, C]: metric to use for similarity
     - w: image width in tokens
     - h: image height in tokens
     - sx: stride in the x dimension for dst, must divide w
     - sy: stride in the y dimension for dst, must divide h
     - r: number of tokens to remove (by merging)
     - no_rand: if true, disable randomness (use top left corner only)
     - rand_seed: if no_rand is false, and if not None, sets random seed.
    """
    B, N, _ = metric.shape

    if r <= 0:
        return do_nothing, do_nothing

    gather = mps_gather_workaround if metric.device.type == "mps" else torch.gather
    
    with torch.no_grad():
        hsy, wsx = h // sy, w // sx

        # For each sy by sx kernel, randomly assign one token to be dst and the rest src
        if no_rand:
            rand_idx = torch.zeros(hsy, wsx, 1, device=metric.device, dtype=torch.int64)
        else:
            rand_idx = torch.randint(sy*sx, size=(hsy, wsx, 1), device=generator.device, generator=generator).to(metric.device)
        
        # The image might not divide sx and sy, so we need to work on a view of the top left if the idx buffer instead
        idx_buffer_view = torch.zeros(hsy, wsx, sy*sx, device=metric.device, dtype=torch.int64)
        idx_buffer_view.scatter_(dim=2, index=rand_idx, src=-torch.ones_like(rand_idx, dtype=rand_idx.dtype))
        idx_buffer_view = idx_buffer_view.view(hsy, wsx, sy, sx).transpose(1, 2).reshape(hsy * sy, wsx * sx)

        # Image is not divisible by sx or sy so we need to move it into a new buffer
        if (hsy * sy) < h or (wsx * sx) < w:
            idx_buffer = torch.zeros(h, w, device=metric.device, dtype=torch.int64)
            idx_buffer[:(hsy * sy), :(wsx * sx)] = idx_buffer_view
        else:
            idx_buffer = idx_buffer_view

        # We set dst tokens to be -1 and src to be 0, so an argsort gives us dst|src indices
        rand_idx = idx_buffer.reshape(1, -1, 1).argsort(dim=1)

        # We're finished with these
        del idx_buffer, idx_buffer_view

        # rand_idx is currently dst|src, so split them
        num_dst = hsy * wsx
        a_idx = rand_idx[:, num_dst:, :] # src
        b_idx = rand_idx[:, :num_dst, :] # dst

        def split(x):
            C = x.shape[-1]
            src = gather(x, dim=1, index=a_idx.expand(B, N - num_dst, C))
            dst = gather(x, dim=1, index=b_idx.expand(B, num_dst, C))
            return src, dst

        # Cosine similarity between A and B
        metric = metric / metric.norm(dim=-1, keepdim=True)
        a, b = split(metric)
        scores = a @ b.transpose(-1, -2)

        # Can't reduce more than the # tokens in src
        r = min(a.shape[1], r)

        # Find the most similar greedily
        node_max, node_idx = scores.max(dim=-1)
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

        unm_idx = edge_idx[..., r:, :]  # Unmerged Tokens
        src_idx = edge_idx[..., :r, :]  # Merged Tokens
        dst_idx = gather(node_idx[..., None], dim=-2, index=src_idx)

    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        src, dst = split(x)
        n, t1, c = src.shape
        
        unm = gather(src, dim=-2, index=unm_idx.expand(n, t1 - r, c))
        src = gather(src, dim=-2, index=src_idx.expand(n, r, c))
        dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce=mode)

        return torch.cat([unm, dst], dim=1)

    def prune(x: torch.Tensor, mode="mean") -> torch.Tensor:
        src, dst = split(x)
        n, t1, c = src.shape
        
        unm = gather(src, dim=-2, index=unm_idx.expand(n, t1 - r, c))

        return torch.cat([unm, dst], dim=1)
    
    def unmerge(x: torch.Tensor) -> torch.Tensor:
        unm_len = unm_idx.shape[1]
        unm, dst = x[..., :unm_len, :], x[..., unm_len:, :]
        _, _, c = unm.shape

        src = gather(dst, dim=-2, index=dst_idx.expand(B, r, c))

        # Combine back to the original shape
        out = torch.zeros(B, N, c, device=x.device, dtype=x.dtype)
        out.scatter_(dim=-2, index=b_idx.expand(B, num_dst, c), src=dst)
        out.scatter_(dim=-2, index=gather(a_idx.expand(B, a_idx.shape[1], 1), dim=1, index=unm_idx).expand(B, unm_len, c), src=unm)
        out.scatter_(dim=-2, index=gather(a_idx.expand(B, a_idx.shape[1], 1), dim=1, index=src_idx).expand(B, r, c), src=src)
        return out
    
    return merge, unmerge, prune


def compute_merge(x: torch.Tensor, tome_info: Dict[str, Any]) -> Tuple[Callable, ...]:
    original_h, original_w = tome_info["size"]
    original_tokens = original_h * original_w
    downsample = int(math.ceil(math.sqrt(original_tokens // x.shape[1])))
    args = tome_info["args"]

    if downsample <= args["max_downsample"]:
        
        w = int(math.ceil(original_w / downsample))
        h = int(math.ceil(original_h / downsample))
        r = int(x.shape[1] * args["ratio"])

        # Re-init the generator if it hasn't already been initialized or device has changed.
        if args["generator"] is None:
            args["generator"] = init_generator(x.device)
        elif args["generator"].device != x.device:
            args["generator"] = init_generator(x.device, fallback=args["generator"])
        
        # If the batch size is odd, then it's not possible for prompted and unprompted images to be in the same
        # batch, which causes artifacts with use_rand, so force it to be off.
        use_rand = False if x.shape[0] % 2 == 1 else args["use_rand"]
        m, u, p = bipartite_soft_matching_random2d(x, w, h, args["sx"], args["sy"], r, 
                                                      no_rand=not use_rand, generator=args["generator"])
    else:
        m, u = (do_nothing, do_nothing)

    if args["prune_replace"] and args["step"] >= args["replace_step"]:
        m_a, u_a = (p, u) if args["merge_attn"]      else (do_nothing, do_nothing)
        m_c, u_c = (p, u) if args["merge_crossattn"] else (do_nothing, do_nothing)
        m_m, u_m = (p, u) if args["merge_mlp"]       else (do_nothing, do_nothing)
    else:
        m_a, u_a = (m, u) if args["merge_attn"]      else (do_nothing, do_nothing)
        m_c, u_c = (m, u) if args["merge_crossattn"] else (do_nothing, do_nothing)
        m_m, u_m = (m, u) if args["merge_mlp"]       else (do_nothing, do_nothing)

    return m_a, m_c, m_m, u_a, u_c, u_m  # Okay this is probably not very good


def make_tome_block(block_class: Type[torch.nn.Module]) -> Type[torch.nn.Module]:
    """
    Make a patched class on the fly so we don't have to import any specific modules.
    This patch applies ToMe to the forward function of the block.
    """

    class ToMeBlock(block_class):
        # Save for unpatching later
        _parent = block_class

        def _forward(self, x: torch.Tensor, context: torch.Tensor = None) -> torch.Tensor:
            m_a, m_c, m_m, u_a, u_c, u_m = compute_merge(x, self._tome_info)

            # This is where the meat of the computation happens
            x = u_a(self.attn1(m_a(self.norm1(x)), context=context if self.disable_self_attn else None)) + x
            x = u_c(self.attn2(m_c(self.norm2(x)), context=context)) + x
            x = u_m(self.ff(m_m(self.norm3(x)))) + x

            return x
    
    return ToMeBlock


def make_diffusers_tome_block(block_class: Type[torch.nn.Module]) -> Type[torch.nn.Module]:
    """
    Make a patched class for a diffusers model.
    This patch applies ToMe to the forward function of the block.
    """
    class ToMeBlock(block_class):
        # Save for unpatching later
        _parent = block_class

        def forward(
            self,
            hidden_states,
            attention_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            timestep=None,
            cross_attention_kwargs=None,
            class_labels=None,
        ) -> torch.Tensor:
            # (1) ToMe
            m_a, m_c, m_m, u_a, u_c, u_m = compute_merge(hidden_states, self._tome_info)

            if self.use_ada_layer_norm:
                norm_hidden_states = self.norm1(hidden_states, timestep)
            elif self.use_ada_layer_norm_zero:
                norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
                    hidden_states, timestep, class_labels, hidden_dtype=hidden_states.dtype
                )
            else:
                norm_hidden_states = self.norm1(hidden_states)

            # (2) ToMe m_a
            norm_hidden_states = m_a(norm_hidden_states)

            # 1. Self-Attention
            cross_attention_kwargs = cross_attention_kwargs if cross_attention_kwargs is not None else {}
            attn_output = self.attn1(
                norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
                attention_mask=attention_mask,
                **cross_attention_kwargs,
            )
            if self.use_ada_layer_norm_zero:
                attn_output = gate_msa.unsqueeze(1) * attn_output

            # (3) ToMe u_a
            hidden_states = u_a(attn_output) + hidden_states

            if self.attn2 is not None:
                norm_hidden_states = (
                    self.norm2(hidden_states, timestep) if self.use_ada_layer_norm else self.norm2(hidden_states)
                )
                # (4) ToMe m_c
                norm_hidden_states = m_c(norm_hidden_states)

                # 2. Cross-Attention
                attn_output = self.attn2(
                    norm_hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=encoder_attention_mask,
                    **cross_attention_kwargs,
                )
                # (5) ToMe u_c
                hidden_states = u_c(attn_output) + hidden_states

            # 3. Feed-forward
            norm_hidden_states = self.norm3(hidden_states)
            
            if self.use_ada_layer_norm_zero:
                norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

            # (6) ToMe m_m
            norm_hidden_states = m_m(norm_hidden_states)

            ff_output = self.ff(norm_hidden_states)

            if self.use_ada_layer_norm_zero:
                ff_output = gate_mlp.unsqueeze(1) * ff_output

            # (7) ToMe u_m
            hidden_states = u_m(ff_output) + hidden_states

            return hidden_states

    return ToMeBlock


def average_cosine_similarity(hidden_states):
    # 归一化hidden_states，以便于计算余弦相似度
    normed_vectors = F.normalize(hidden_states, p=2, dim=2)
    
    # 计算余弦相似度矩阵
    cosine_sim_matrix = torch.matmul(normed_vectors, normed_vectors.transpose(1, 2))
    
    # 仅考虑上三角部分，避免自比较和重复计算
    upper_tri_indices = torch.triu_indices(cosine_sim_matrix.size(1), cosine_sim_matrix.size(2), offset=1)
    average_similarity = cosine_sim_matrix[:, upper_tri_indices[0], upper_tri_indices[1]].mean()
    
    return average_similarity




# def show_timing_info():
#     # 创建一个列表，包含所有时间数据和相应的标签
#     data = [
#         ["Normalization 1", norm1],
#         ["Compute Merge", compute_merge],
#         ["Context Normalization", contextnorm],
#         ["Merge Attention", merge_attn],
#         ["Attention", attn],
#         ["Unmerge Attention", unmerge_attn],
#         ["Normalization 2", norm2],
#         ["Merge MLP", merge_mlp],
#         ["MLP", mlp],
#         ["Unmerge MLP", unmerge_mlp]
#     ]

#     # 使用 tabulate 打印表格，选择 "grid" 格式使表格更易读
#     print(tabulate(data, headers=['Operation', 'Time (seconds)'], tablefmt='grid'))

def show_timing_info():
    # 创建一个列表，包含所有时间数据和相应的标签
    data = [
        ["Norm", time_norm1 + time_norm2],
        ["Norm Context", time_contextnorm],
        ["Compute Merge", time_compute_merge],
        ["Merge Attention", time_merge_attn],
        ["Attention", time_attn],
        ["Unmerge Attention", time_unmerge_attn],
        ["Normalization 2", time_norm2],
        ["Merge MLP", time_merge_mlp],
        ["MLP", time_mlp],
        ["Unmerge MLP", time_unmerge_mlp]
    ]

    # 使用 tabulate 打印表格，选择 "grid" 格式使表格更易读
    print(tabulate(data, headers=['Operation', 'Time (seconds)'], tablefmt='grid'))

def make_diffusers_tome_block_mmdit(block_class: Type[torch.nn.Module]) -> Type[torch.nn.Module]:
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
            global time_norm1, time_compute_merge, time_contextnorm, time_merge_attn, time_attn, time_unmerge_attn, \
                    time_norm2, time_merge_mlp, time_mlp, time_unmerge_mlp
            start_time = time.time()
            norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, emb=temb)
            time_norm1 += time.time() - start_time
            
            # (1) ToMe
            start_time = time.time()
            m_a, _, m_m, u_a, _, u_m = compute_merge(norm_hidden_states, self._tome_info)
            time_compute_merge += time.time() - start_time

            start_time = time.time()
            if self.context_pre_only:
                norm_encoder_hidden_states = self.norm1_context(encoder_hidden_states, temb)
            else:
                norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(
                    encoder_hidden_states, emb=temb
                )
            time_contextnorm += time.time() - start_time

            # (2) ToMe m_a
            start_time = time.time()
            norm_hidden_states = m_a(norm_hidden_states)
            time_merge_attn += time.time() - start_time
            
            # Attention.
            start_time = time.time()
            attn_output, context_attn_output = self.attn(
                hidden_states=norm_hidden_states, encoder_hidden_states=norm_encoder_hidden_states
            )
            # Process attention outputs for the `hidden_states`.
            attn_output = gate_msa.unsqueeze(1) * attn_output
            time_attn += time.time() - start_time

            # (3) ToMe u_a
            start_time = time.time()
            attn_output = u_a(attn_output)
            time_unmerge_attn += time.time() - start_time
            hidden_states = hidden_states + attn_output

            start_time = time.time()
            norm_hidden_states = self.norm2(hidden_states)
            norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
            time_norm2 += time.time() - start_time

            # (4) ToMe m_m
            start_time = time.time()
            norm_hidden_states = m_m(norm_hidden_states)
            time_merge_mlp += time.time() - start_time

            start_time = time.time()
            if self._chunk_size is not None:
                # "feed_forward_chunk_size" can be used to save memory
                ff_output = _chunked_feed_forward(self.ff, norm_hidden_states, self._chunk_dim, self._chunk_size)
            else:
                ff_output = self.ff(norm_hidden_states)
            ff_output = gate_mlp.unsqueeze(1) * ff_output
            time_mlp += time.time() - start_time

            # (5) ToMe u_m
            start_time = time.time()
            ff_output = u_m(ff_output)
            time_unmerge_mlp += time.time() - start_time

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
            time_contextnorm += time.time() - start_time
            return encoder_hidden_states, hidden_states

    return JointTransformerBlock_ToMe

def hook_tome_model(model: torch.nn.Module):
    """ Adds a forward pre hook to get the image size. This hook can be removed with remove_patch. """
    def hook(module, args):
        if isinstance_str(module, "SD3Transformer2DModel"):
            pass
        else:
            module._tome_info["size"] = (args[0].shape[2], args[0].shape[3])
        return None

    model._tome_info["hooks"].append(model.register_forward_pre_hook(hook))


def apply_patch_tome(
        model: torch.nn.Module,
        ratio: float = 0.5,
        max_downsample: int = 1,
        sx: int = 2, sy: int = 2,
        use_rand: bool = True,
        merge_attn: bool = True,
        merge_crossattn: bool = False,
        merge_mlp: bool = False,
        ratio_start: float = 0,
        ratio_end: float = 0,
        save_dir: str = None,
        prune_replace: bool = False,
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
    remove_patch(model)

    is_diffusers = isinstance_str(model, "DiffusionPipeline") or isinstance_str(model, "ModelMixin")

    if not is_diffusers:
        if not hasattr(model, "model") or not hasattr(model.model, "diffusion_model"):
            # Provided model not supported
            raise RuntimeError("Provided model was not a Stable Diffusion / Latent Diffusion model, as expected.")
        diffusion_model = model.model.diffusion_model
    else:
        # Supports "pipe.unet" and "pipe.transformer" and "unet" and "transformer" 
        diffusion_model = model.unet if hasattr(model, "unet") else model.transformer if hasattr(model, "transformer") else model

    diffusion_model._tome_info = {
        "hooks": [],
        "size": (),
        "save_dir" : save_dir,
        "args": {
            "step": 0,
            "ratio_start": ratio_start,
            "ratio_end": ratio_end,
            "ratio": ratio,
            "step_ratio": ratio,
            "max_downsample": max_downsample,
            "sx": sx, "sy": sy,
            "use_rand": use_rand,
            "generator": None,
            "merge_attn": merge_attn,
            "merge_crossattn": merge_crossattn,
            "merge_mlp": merge_mlp,
            "prune_replace": prune_replace,
            "replace_step": replace_step,
        }
    }
    hook_tome_model(diffusion_model)

    for _, module in diffusion_model.named_modules():
        # If for some reason this has a different name, create an issue and I'll fix it
        if isinstance_str(module, "BasicTransformerBlock"):
            make_tome_block_fn = make_diffusers_tome_block if is_diffusers else make_tome_block
            module.__class__ = make_tome_block_fn(module.__class__)
            module._tome_info = diffusion_model._tome_info

            # Something introduced in SD 2.0 (LDM only)
            if not hasattr(module, "disable_self_attn") and not is_diffusers:
                module.disable_self_attn = False

            # Something needed for older versions of diffusers
            if not hasattr(module, "use_ada_layer_norm_zero") and is_diffusers:
                module.use_ada_layer_norm = False
                module.use_ada_layer_norm_zero = False
        if isinstance_str(module, "JointTransformerBlock"):
            make_tome_block_fn = make_diffusers_tome_block_mmdit if is_diffusers else make_tome_block
            module.__class__ = make_tome_block_fn(module.__class__)
            module._tome_info = diffusion_model._tome_info

    return model


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


# def make_diffusers_tome_block_mmdit(block_class: Type[torch.nn.Module]) -> Type[torch.nn.Module]:
#     """
#     Make a patched class for a diffusers model.
#     This patch applies ToMe to the forward function of the block.
#     """
#     class JointTransformerBlock_ToMe(block_class):
#         # Save for unpatching later
#         _parent = block_class

#         def forward(
#             self, hidden_states: torch.FloatTensor, encoder_hidden_states: torch.FloatTensor, temb: torch.FloatTensor
#         ):  
            
#             # # 计算初始hidden_states的平均余弦相似度
#             # average_sim = average_cosine_similarity(hidden_states)
            
#             # # 将相似度写入文件
#             # with open("/data1/fanghaipeng/project/sora/tomesd/tomesd/average_cosine_similarity.txt", "a") as file:
#             #     file.write(f"{average_sim.item()}\n")
#             norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, emb=temb)
            
#             # (1) ToMe
#             m_a, _, m_m, u_a, _, u_m = compute_merge(norm_hidden_states, self._tome_info)
#             if self.context_pre_only:
#                 norm_encoder_hidden_states = self.norm1_context(encoder_hidden_states, temb)
#             else:
#                 norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(
#                     encoder_hidden_states, emb=temb
#                 )

#             # (2) ToMe m_a
            
#             norm_hidden_states = m_a(norm_hidden_states)

#             # Attention.
#             attn_output, context_attn_output = self.attn(
#                 hidden_states=norm_hidden_states, encoder_hidden_states=norm_encoder_hidden_states
#             )

#             # Process attention outputs for the `hidden_states`.
#             attn_output = gate_msa.unsqueeze(1) * attn_output

#             # (3) ToMe u_a
#             hidden_states = hidden_states + u_a(attn_output)
            
#             norm_hidden_states = self.norm2(hidden_states)
#             norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

#             # (4) ToMe m_m
#             norm_hidden_states = m_m(norm_hidden_states)

#             if self._chunk_size is not None:
#                 # "feed_forward_chunk_size" can be used to save memory
#                 ff_output = _chunked_feed_forward(self.ff, norm_hidden_states, self._chunk_dim, self._chunk_size)
#             else:
#                 ff_output = self.ff(norm_hidden_states)
#             ff_output = gate_mlp.unsqueeze(1) * ff_output

#             # (5) ToMe u_m
#             hidden_states = hidden_states + u_m(ff_output)

#             # Process attention outputs for the `encoder_hidden_states`.
#             if self.context_pre_only:
#                 encoder_hidden_states = None
#             else:
#                 context_attn_output = c_gate_msa.unsqueeze(1) * context_attn_output
#                 encoder_hidden_states = encoder_hidden_states + context_attn_output

#                 norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
#                 norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]
#                 if self._chunk_size is not None:
#                     # "feed_forward_chunk_size" can be used to save memory
#                     context_ff_output = _chunked_feed_forward(
#                         self.ff_context, norm_encoder_hidden_states, self._chunk_dim, self._chunk_size
#                     )
#                 else:
#                     context_ff_output = self.ff_context(norm_encoder_hidden_states)
#                 encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output

#             return encoder_hidden_states, hidden_states

#     return JointTransformerBlock_ToMe
import torch
from typing import Tuple, Callable
import math
from typing import Type, Dict, Any, Tuple, Callable
import torch.nn.functional as F
from diffusers.models.attention import _chunked_feed_forward
import random
import time
from tabulate import tabulate
from einops import rearrange

def average_cosine_similarity(hidden_states):
    # 归一化hidden_states，以便于计算余弦相似度
    normed_vectors = F.normalize(hidden_states, p=2, dim=2)
    
    # 计算余弦相似度矩阵
    cosine_sim_matrix = torch.matmul(normed_vectors, normed_vectors.transpose(1, 2))
    
    # 仅考虑上三角部分，避免自比较和重复计算
    upper_tri_indices = torch.triu_indices(cosine_sim_matrix.size(1), cosine_sim_matrix.size(2), offset=1)
    average_similarity = cosine_sim_matrix[:, upper_tri_indices[0], upper_tri_indices[1]].mean()
    
    return average_similarity

def do_nothing(x: torch.Tensor, mode:str=None):
    return x

def mps_gather_workaround(input, dim, index):
    if input.shape[-1] == 1:
        return torch.gather(
            input.unsqueeze(-1),
            dim - 1 if dim < 0 else dim,
            index.unsqueeze(-1)
        ).squeeze(-1)
    else:
        return torch.gather(input, dim, index)
    

def merge_wavg(
    merge: Callable, x: torch.Tensor, size: torch.Tensor = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    
    if size is None:
        size = torch.ones_like(x[..., 0, None])

    x = merge(x * size, mode="sum")
    size = merge(size, mode="sum")    
    x = x / size
    
    return x, size

def merge_source(
    merge: Callable, x: torch.Tensor, source: torch.Tensor = None
) -> torch.Tensor:

    if source is None:
        n, t, _ = x.shape
        source = torch.eye(t, device=x.device)[None, ...].expand(n, t, t)

    source = merge(source, mode="amax")
    return source

def global_merge_1d(
        metric: torch.Tensor,
        layer_idx: int = 0,
        reduce_num: int = 0,
        no_rand: bool = False,
        generator: torch.Generator = None) -> Tuple[Callable, Callable]:
    
    t = metric.shape[1]

    if reduce_num <= 0:
        return do_nothing, do_nothing, do_nothing
    # print(r)
    gather = mps_gather_workaround if metric.device.type == "mps" else torch.gather

    with torch.no_grad():

        metric = metric / metric.norm(dim=-1, keepdim=True)
        B, N, C = metric.shape
        num_dst = N // 4

        if no_rand:
            all_indices = torch.arange(N, device=generator.device).unsqueeze(0).unsqueeze(-1).expand(B, N, 1)
            a_idx = torch.cat((all_indices[:, 1::4, :], all_indices[:, 2::4, :], all_indices[:, 3::4, :]), dim=1)
            b_idx = all_indices[:, ::4, :]
        else:
            all_indices = torch.rand(B, N, generator=generator, device=generator.device).argsort(dim=1).unsqueeze(-1)
            a_idx = all_indices[:,num_dst:,:]
            b_idx = all_indices[:,:num_dst,:]

        def split(x):
            c = x.shape[-1]
            src = gather(x, dim=1, index=a_idx.expand(B, N - num_dst, c))
            dst = gather(x, dim=1, index=b_idx.expand(B, num_dst, c))
            return src, dst
        
        a, b = split(metric)
        scores = a @ b.transpose(-1, -2)

        node_max, node_idx = scores.max(dim=-1)
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

        # ------------------ start  addaptive section --------- 
        if reduce_num is None: # not design
            i = layer_idx
            n_B, n_H = node_max.shape
            node_mean= torch.add(node_max[:,1:].mean(dim=1).mean(),node_max[:,1:].std(dim=1).mean()/i)
            node_mean=node_mean.repeat(1,n_H)
            reduce_num = torch.ge(node_max, node_mean).sum(dim=1).min()
        else: 
            reduce_num = min(a.shape[1], reduce_num)

        # ------------------ end addaptive section --------- 
        
        unm_idx = edge_idx[..., reduce_num:, :]  # Unmerged Tokens
        src_idx = edge_idx[..., :reduce_num, :]  # Merged Tokens
        dst_idx = gather(node_idx[..., None], dim=-2, index=src_idx)

    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        a, dst = split(x)
        n, t1, c = a.shape
        unm = gather(a, dim=-2, index=unm_idx.expand(n, t1 - reduce_num, c)) 
        src = gather(a, dim=-2, index=src_idx.expand(n, reduce_num, c))
        dst = dst.scatter_reduce(-2, dst_idx.expand(n, reduce_num, c), src, reduce=mode)

        return torch.cat([unm, dst], dim=1)
    
    def mprune(x: torch.Tensor, mode="mean") -> torch.Tensor:

        a, dst = split(x)
        n, t1, c = a.shape

        unm = gather(a, dim=-2, index=unm_idx.expand(n, t1 - reduce_num, c))

        return torch.cat([unm, dst], dim=1)

    def unmerge(x: torch.Tensor) -> torch.Tensor:
        unm_len = unm_idx.shape[1]
        unm, dst = x[..., :unm_len, :], x[..., unm_len:, :]
        _, _, c = unm.shape

        src = gather(dst, dim=-2, index=dst_idx.expand(B, reduce_num, c))

        # Combine back to the original shape
        out = torch.zeros(B, N, c, device=x.device, dtype=x.dtype)
        # NOTE: a_idx is (a in x) b_idx is (dst in x), 
        # NOTE: dst_idx is (src in dst), unm_idx is (unm in a), (src_idx) is (src in a)
        out.scatter_(dim=-2, index=b_idx.expand(B, num_dst, c), src=dst) 
        out.scatter_(dim=-2, index=gather(a_idx.expand(B, a_idx.shape[1], 1), dim=1, index=unm_idx).expand(B, unm_len, c), src=unm) 
        out.scatter_(dim=-2, index=gather(a_idx.expand(B, a_idx.shape[1], 1), dim=1, index=src_idx).expand(B, reduce_num, c), src=src)
        return out
    
    return merge, mprune, unmerge

def global_merge_2d(
        metric: torch.Tensor,
        w: int, h: int, sx: int, sy: int, 
        reduce_num: int,
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

    if reduce_num <= 0:
        return do_nothing, do_nothing, do_nothing

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
        reduce_num = min(a.shape[1], reduce_num)

        # Find the most similar greedily
        node_max, node_idx = scores.max(dim=-1)
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

        unm_idx = edge_idx[..., reduce_num:, :]  # Unmerged Tokens
        src_idx = edge_idx[..., :reduce_num, :]  # Merged Tokens
        dst_idx = gather(node_idx[..., None], dim=-2, index=src_idx)

    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        a, dst = split(x)
        n, t1, c = src.shape
        
        unm = gather(a, dim=-2, index=unm_idx.expand(n, t1 - reduce_num, c))
        src = gather(a, dim=-2, index=src_idx.expand(n, reduce_num, c))
        dst = dst.scatter_reduce(-2, dst_idx.expand(n, reduce_num, c), src, reduce=mode)

        return torch.cat([unm, dst], dim=1)

    def mprune(x: torch.Tensor) -> torch.Tensor:
        a, dst = split(x)
        n, t1, c = a.shape
        
        unm = gather(a, dim=-2, index=unm_idx.expand(n, t1 - reduce_num, c))

        return torch.cat([unm, dst], dim=1)
    
    def unmerge(x: torch.Tensor) -> torch.Tensor:
        unm_len = unm_idx.shape[1]
        unm, dst = x[..., :unm_len, :], x[..., unm_len:, :]
        _, _, c = unm.shape

        src = gather(dst, dim=-2, index=dst_idx.expand(B, reduce_num, c))

        # Combine back to the original shape
        out = torch.zeros(B, N, c, device=x.device, dtype=x.dtype)
        # NOTE: a_idx is (a in x) b_idx is (dst in x), 
        # NOTE: dst_idx is (src in dst), unm_idx is (unm in a), (src_idx) is (src in a)
        out.scatter_(dim=-2, index=b_idx.expand(B, num_dst, c), src=dst)
        out.scatter_(dim=-2, index=gather(a_idx.expand(B, a_idx.shape[1], 1), dim=1, index=unm_idx).expand(B, unm_len, c), src=unm)
        out.scatter_(dim=-2, index=gather(a_idx.expand(B, a_idx.shape[1], 1), dim=1, index=src_idx).expand(B, reduce_num, c), src=src)
        return out
    
    return merge, mprune, unmerge


def local_merge_2d(
    metric: torch.Tensor,
    reduce_num: int = 0,
    threshold: float = 0,
    window_size: Tuple[int, int] = (2,2),
    no_rand: bool = False,
    generator: torch.Generator = None
) -> Tuple[Callable, Callable]:
    
    if reduce_num <= 0:
        return do_nothing, do_nothing, do_nothing

    gather = mps_gather_workaround if metric.device.type == "mps" else torch.gather

    with torch.no_grad():
        
        ws_h, ws_w = int(window_size[0]), int(window_size[1])
        stride_h, stride_w = ws_h, ws_w
        num_token_window = stride_h * stride_w
        
        B, N, D = metric.size()
        base_grid_H = int(math.sqrt(N))
        base_grid_W = base_grid_H
        assert base_grid_H * base_grid_W == N and base_grid_H % ws_h == 0 and base_grid_W % ws_w == 0

        metric = rearrange(metric, "b (h w) c -> b c h w", h=base_grid_H)
    
        metric = rearrange(metric, 'b c (gh ps_h) (gw ps_w) -> b gh gw c ps_h ps_w', gh=base_grid_H//ws_h, gw=base_grid_W//ws_w)
        b, gh, gw, c, ps_h, ps_w = metric.shape

        # Flatten mxm window for pairwise operations
        tensor_flattened = metric.reshape(b, gh, gw, c, -1)
    

        # Expand dims for pairwise operations
        tensor_1 = tensor_flattened.unsqueeze(-1)
        tensor_2 = tensor_flattened.unsqueeze(-2)

        # Compute cosine similarities
        sims = F.cosine_similarity(tensor_1, tensor_2, dim=3)

        # Exclude the self-similarity (i.e., similarity with oneself will be 1)
        sims_mask = 1 - torch.eye(ps_h * ps_w).to(sims.device)
        sims = sims * sims_mask

        # Average similarities (excluding the self-similarity)
        similarity_map = sims.sum(-1).sum(-1) / ((ps_h * ps_w) * (ps_h * ps_w - 1))
            
        similarity_map = rearrange(similarity_map.unsqueeze(1), 'b c h w-> b (c h w)')
        
        #--- adaptive section ---#
        if reduce_num is None:
            n_B, n_H = similarity_map.shape
            node_mean = torch.tensor(threshold).cuda(sims.device)
            node_mean=node_mean.repeat(1,n_H)
            reduce_num = torch.ge(similarity_map, node_mean).sum(dim=1).min()
        else:
            reduce_num = reduce_num // 3

        # -------------# 
    
        #   get top k similar super patches 
        _, sim_super_patch_idxs = similarity_map.topk(reduce_num, dim=-1)
    
        # --- creating the mergabel and unmergable super  pathes
        tensor = torch.arange(base_grid_H * base_grid_W).reshape(base_grid_H, base_grid_W).to(metric.device)

        # Repeat the tensor to create a batch of size 2
        tensor = tensor.unsqueeze(0).repeat(B, 1, 1)
        

        # Apply unfold operation on last two dimensions to create the sliding window
        windowed_tensor = tensor.unfold(1, ws_h, stride_h).unfold(2, ws_w, stride_w)

        # Reshape the tensor to the desired shape 
        windowed_tensor = windowed_tensor.reshape(B, -1, num_token_window)
    
        # Use torch.gather to collect the desired elements
        gathered_tensor = torch.gather(windowed_tensor, 1, sim_super_patch_idxs.unsqueeze(-1).expand(-1, -1, num_token_window))


        # Create a mask for all indices, for each batch
        mask = torch.ones((B, windowed_tensor.shape[1]), dtype=bool).to(metric.device)

        # Create a tensor that matches the shape of indices and fill it with False
        mask_values = torch.zeros_like(sim_super_patch_idxs, dtype=torch.bool).to(metric.device)

        # Use scatter_ to update the mask. This will set mask[b, indices[b]] = False for all b
        mask.scatter_(1, sim_super_patch_idxs, mask_values)

        # Get the remaining tensor
        remaining_tensor = windowed_tensor[mask.unsqueeze(-1).expand(-1, -1, num_token_window)].reshape(B, -1, num_token_window)
        unm_idx = remaining_tensor.reshape(B, -1).sort(dim=-1).values.unsqueeze(-1)
        dim_index = (num_token_window)- 1 
        src_idx= gathered_tensor[:, :, :dim_index].reshape(B, -1).unsqueeze(-1)
        dst_idx= gathered_tensor[:, :, dim_index].reshape(B, -1).unsqueeze(-1)
        merge_idx = torch.arange(src_idx.shape[1]//dim_index).repeat_interleave(dim_index).repeat(B, 1).unsqueeze(-1).to(metric.device)


    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
       # TODO: num_token_window can be undefined
       
    
        n, t1, c = x.shape
        # src = x.gather(dim=-2, index=src_idx.expand(n, r*dim_index, c))
        # dst = x.gather(dim=-2, index=dst_idx.expand(n, r, c))
        # unm = x.gather(dim=-2, index=unm_idx.expand(n, t1 - (r*num_token_window), c))
        src = gather(x, dim=-2, index=src_idx.expand(n, reduce_num*dim_index, c))
        dst = gather(x, dim=-2, index=dst_idx.expand(n, reduce_num, c))
        unm = gather(x, dim=-2, index=unm_idx.expand(n, t1 - (reduce_num*num_token_window), c))
        dst = dst.scatter_reduce(-2, merge_idx.expand(n,reduce_num*dim_index, c), src, reduce=mode)
        x = torch.cat([unm, dst], dim=1)

        return x
    
    def mprune(x: torch.Tensor, mode="mean") -> torch.Tensor:
       # TODO: num_token_window can be undefined
        n, t1, c = x.shape

        dst = gather(x, dim=-2, index=dst_idx.expand(n, reduce_num, c))
        unm = gather(x, dim=-2, index=unm_idx.expand(n, t1 - (reduce_num*num_token_window), c))
        x = torch.cat([unm, dst], dim=1)

        return x

    def unmerge(x: torch.Tensor) -> torch.Tensor:
        unm_len = unm_idx.shape[1]
        unm, dst = x[..., :unm_len, :], x[..., unm_len:, :]
        _, tu, c = unm.shape

        src = gather(dst, dim=-2, index=dst_idx.expand(B, reduce_num, c))

        # Combine back to the original shape
        out = torch.zeros(B, N, c, device=x.device, dtype=x.dtype)
        # NOTE: src_idx is (src in x), dst_idx is (dst in x), unm_idx is (unm in x)
        out.scatter_(dim=-2, index=dst_idx.expand(B, reduce_num, c), src=dst)
        out.scatter_(dim=-2, index=unm_idx.expand(B, tu, c), src=unm)
        out.scatter_(dim=-2, index=src_idx.expand(B, reduce_num*dim_index, c), src=src)
        return out

    return merge, mprune, unmerge
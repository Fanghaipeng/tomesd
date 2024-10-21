import torch
from typing import Tuple, Callable
import math
import torch.nn.functional as F
import random

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

    return merge, unmerge

def soft_interpolate(metric: torch.Tensor,
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
    B, N, C = metric.shape

    if r <= 0:
        return do_nothing, do_nothing

    assert N == w * h, "The total number of tokens (N) must equal w * h"

    # Calculate new dimensions
    new_area = (1 - r) * h * w
    new_w = int(math.sqrt(new_area * (w / h)))
    new_h = int(new_area / new_w)
    print(new_h)
    print(new_w)
    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        x = x.view(B, h, w, C)
        x = x.permute(0, 3, 1, 2)
        x_interpolate = F.interpolate(x, size=(new_h, new_w), mode="bilinear", align_corners=False)
        x_out = x_interpolate.permute(0, 2, 3, 1).view(B, -1, C)
        return x_out

    def unmerge(x: torch.Tensor) -> torch.Tensor:
        x = x.view(B, new_h, new_w, C)
        x = x.permute(0, 3, 1, 2)
        x_interpolate = F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)
        x_out = x_interpolate.permute(0, 2, 3, 1).view(B, -1, C)
        return x_out

    return merge, unmerge

def clamp(value, min_value, max_value):
    return max(min_value, min(value, max_value))

def calculate_indices(windowsizei, windowsizej, h, w, new_h, new_w, x_device, random_select=True):
    max_area_coverage = torch.zeros((h, w), device=x_device)
    merge_indices = torch.full((new_h, new_w, 2), -1, device=x_device)
    unmerge_indices = torch.full((h, w, 2), -1, device=x_device)

    for new_i in range(new_h):
        for new_j in range(new_w):
            windowarea_start_i = new_i * windowsizei
            windowarea_start_j = new_j * windowsizej
            windowarea_end_i = windowarea_start_i + windowsizei
            windowarea_end_j = windowarea_start_j + windowsizej

            index_start_i = math.floor(windowarea_start_i)
            index_start_j = math.floor(windowarea_start_j)
            index_end_i = math.ceil(windowarea_end_i)
            index_end_j = math.ceil(windowarea_end_j)

            weights = []
            indices = []

            # i 行 j 列
            for old_i in range(index_start_i, index_end_i):
                for old_j in range(index_start_j, index_end_j):
                    left = clamp(old_i, windowarea_start_i, windowarea_end_i)
                    right = clamp(old_i + 1, windowarea_start_i, windowarea_end_i)
                    top = clamp(old_j, windowarea_start_j, windowarea_end_j)
                    bottom = clamp(old_j + 1, windowarea_start_j, windowarea_end_j)
                    area = max(0, right - left) * max(0, bottom - top)
                    if area > 0:
                        weights.append(area)
                        indices.append((old_i, old_j))

            if random_select:
                chosen_index = random.choices(indices, weights=weights, k=1)[0]
            else:
                max_idx = weights.index(max(weights))
                chosen_index = indices[max_idx]        

            merge_indices[new_i, new_j, 0], merge_indices[new_i, new_j, 1] = chosen_index

            for weight, (old_i, old_j) in zip(weights, indices):
                if weight > max_area_coverage[old_i, old_j]:
                    max_area_coverage[old_i, old_j] = weight
                    unmerge_indices[old_i, old_j, 0], unmerge_indices[old_i, old_j, 1] = new_i, new_j

    return merge_indices, unmerge_indices, max_area_coverage

def structural_tokenselect(x: torch.Tensor, w: int, h: int, r: float, no_rand: bool = False, generator: torch.Generator = None) -> Tuple[Callable, Callable]:
    B, N, C = x.shape
    assert N == w * h, "The total number of tokens (N) must equal w * h"

    if r <= 0:
        return lambda x: x, lambda x: x

    new_area = (1 - r) * h * w
    new_w, new_h = int(math.sqrt(new_area * (w / h))), int(new_area / math.sqrt(new_area * (w / h)))
    si, sj = h / new_h, w / new_w

    # merge_indices, unmerge_indices, max_area_coverage = calculate_indices(si, sj, h, w, new_h, new_w, x.device, no_rand=no_rand)
    merge_indices, unmerge_indices, max_area_coverage = calculate_indices(si, sj, h, w, new_h, new_w, x.device, random_select=False)
    # merge_indices, unmerge_indices, max_area_coverage = calculate_indices(si, sj, h, w, new_h, new_w, x.device, random_select=False)
    # merge_indices, unmerge_indices, max_area_coverage = calculate_indices_vectorized(si, sj, h, w, new_h, new_w, x.device)

    merge_indices = merge_indices[..., 0] * w + merge_indices[..., 1]
    merge_indices = merge_indices.view(new_h * new_w, -1).unsqueeze(0).expand(B, -1, C)
    unmerge_indices = unmerge_indices[..., 0] * new_h + unmerge_indices[..., 1]
    unmerge_indices = unmerge_indices.view(h * w, -1).unsqueeze(0).expand(B, -1, C)

    def merge(x: torch.Tensor) -> torch.Tensor:
        out = torch.gather(x, 1, merge_indices)
        return out

    def unmerge(x: torch.Tensor) -> torch.Tensor:
        out = torch.gather(x, 1, unmerge_indices)
        return out

    return merge, unmerge


def calculate_indices_weights(windowsizei, windowsizej, h, w, new_h, new_w, x_dtype, x_device):
    # 准备存储索引和权重的数据结构
    weights = torch.zeros((new_h, new_w, h, w), dtype=x_dtype, device=x_device)  # 存储权重

    for new_i in range(new_h):
        for new_j in range(new_w):
            windowarea_start_i = new_i * windowsizei
            windowarea_start_j = new_j * windowsizej
            windowarea_end_i = windowarea_start_i + windowsizei
            windowarea_end_j = windowarea_start_j + windowsizej

            index_start_i = math.floor(windowarea_start_i)
            index_start_j = math.floor(windowarea_start_j)
            index_end_i = math.ceil(windowarea_end_i)
            index_end_j = math.ceil(windowarea_end_j)

            for old_i in range(index_start_i, index_end_i):
                for old_j in range(index_start_j, index_end_j):
                    left = clamp(old_i, windowarea_start_i, windowarea_end_i)
                    right = clamp(old_i + 1, windowarea_start_i, windowarea_end_i)
                    top = clamp(old_j, windowarea_start_j, windowarea_end_j)
                    bottom = clamp(old_j + 1, windowarea_start_j, windowarea_end_j)
                    area = max(0, right - left) * max(0, bottom - top)

                    if area > 0:
                        weights[new_i, new_j, old_i, old_j] = area

    return weights

def structural_tokenmerge(x: torch.Tensor, w: int, h: int, r: float, no_rand: bool = False, generator: torch.Generator = None) -> Tuple[Callable, Callable]:
    B, N, C = x.shape
    assert N == w * h, "The total number of tokens (N) must equal w * h"

    if r <= 0:
        return lambda x: x, lambda x: x

    new_area = (1 - r) * h * w
    new_w, new_h = int(math.sqrt(new_area * (w / h))), int(new_area / math.sqrt(new_area * (w / h)))
    si, sj = h / new_h, w / new_w

    weights = calculate_indices_weights(si, sj, h, w, new_h, new_w, x.dtype, x.device)

    def merge(x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        new_h, new_w, _, _ = weights.shape
        output = torch.zeros((B, new_h * new_w, C), dtype=x.dtype, device=x.device)

        for new_i in range(new_h):
            for new_j in range(new_w):
                valid_mask = weights[new_i, new_j].view(h * w) > 0
                valid_weights = weights[new_i, new_j].view(h * w)[valid_mask]
                valid_weights = valid_weights / valid_weights.sum()
                weighted_sum = torch.einsum('m,nmc->nc', valid_weights, x[:, valid_mask])
                output[:, new_i * new_w + new_j, :] = weighted_sum

        return output

    def unmerge(x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        _, _, h, w = weights.shape
        output = torch.zeros((B, h * w, C), dtype=x.dtype, device=x.device)

        # #! area unmerge
        # for old_i in range(h):
        #     for old_j in range(w):
        #         valid_mask = weights[:, :, old_i, old_j].view(new_h * new_w) > 0
        #         valid_weights = weights[:, :, old_i, old_j].view(new_h * new_w)[valid_mask]
        #         valid_weights = valid_weights / valid_weights.sum()
        #         weighted_sum = torch.einsum('m,nmc->nc', valid_weights, x[:, valid_mask])
        #         output[:, old_i * w + old_j, :] = weighted_sum

        #! max unmerge
        for old_i in range(h):
            for old_j in range(w):
                # 获取当前位置的权重并找出最大权重的索引
                local_weights = weights[:, :, old_i, old_j].view(new_h * new_w)
                max_idx = torch.argmax(local_weights, dim=0)  # 返回最大元素的索引
                
                # 仅选取最大权重对应的x值
                output[:, old_i * w + old_j, :] = x[:, max_idx, :]

        return output

    return merge, unmerge
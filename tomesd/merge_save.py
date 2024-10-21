def structural_tokenselect_maxarea(x: torch.Tensor, w: int, h: int, r: float, no_rand: bool = False, generator: torch.Generator = None) -> Tuple[Callable, Callable]:
    B, N, C = x.shape
    assert N == w * h, "The total number of tokens (N) must equal w * h"

    if r <= 0:
        return lambda x: x, lambda x: x

    # Calculate new dimensions
    new_area = (1 - r) * h * w
    new_w = int(math.sqrt(new_area * (w / h)))
    new_h = int(new_area / math.sqrt(new_area * (w / h)))

    si = h / new_h
    sj = w / new_w

    # Prepare to track max area coverage and mappings for each original token
    max_area_coverage = torch.zeros((h, w), device=x.device)
    merge_indices = torch.full((new_h, new_w, 2), -1, device=x.device)  # Store indices of new tokens
    unmerge_indices = torch.full((h, w, 2), -1, device=x.device)  # Store indices of new tokens

    def merge(x: torch.Tensor) -> torch.Tensor:
        x = x.view(B, h, w, C)
        new_x = torch.zeros((B, new_h, new_w, C), device=x.device, dtype=x.dtype)

        for i in range(new_h):
            for j in range(new_w):
                start_i = i * si
                start_j = j * sj
                end_i = start_i + si
                end_j = start_j + sj

                istart = int(math.floor(start_i))
                jstart = int(math.floor(start_j))
                iend = min(int(math.ceil(end_i)), h)
                jend = min(int(math.ceil(end_j)), w)

                weights = []
                indices = []

                for ii in range(istart, iend):
                    for jj in range(jstart, jend):
                        top = max(start_i, ii)
                        left = max(start_j, jj)
                        bottom = min(end_i, ii + 1)
                        right = min(end_j, jj + 1)
                        area = max(0, right - left) * max(0, bottom - top)
                        if area > 0:
                            weights.append(area)
                            indices.append((ii, jj))

                # Select the token with the highest area coverage for each window
                if weights:
                    max_idx = weights.index(max(weights))
                    max_ii, max_jj = indices[max_idx]
                    merge_indices[i, j, :] = max_ii, max_jj
                    new_x[:, i, j, :] = x[:, merge_indices[i, j, 0], merge_indices[i, j, 1], :]

                    # Record mapping for unmerge
                    for weight, (ii, jj) in zip(weights, indices):
                        if weight > max_area_coverage[ii, jj]:
                            max_area_coverage[ii, jj] = weight
                            unmerge_indices[ii, jj, :] = torch.tensor([i, j], device=x.device)

        out = new_x.view(B, new_h * new_w, C)

        return out

    def unmerge(x: torch.Tensor) -> torch.Tensor:
        B, N1, C = x.shape
        x = x.view(B, new_h, new_w, C)
        out = torch.zeros((B, h, w, C), device=x.device, dtype=x.dtype)

        # Prepare indices for gathering
        # unmerge_indices should have the shape [B, h, w, 2] and contains indices in the flattened view of [new_h, new_w]
        gather_indices = unmerge_indices[..., 0] * new_h + unmerge_indices[..., 1]  # Convert 2D indices to 1D indices
        gather_indices = gather_indices.view(h * w, -1).unsqueeze(0).expand(B, -1, C)  # Expand indices for all channels

        # Reshape x for gathering
        x_flat = x.view(B, new_h * new_w, C)  # Flatten spatial dimensions for gathering

        # Gather elements based on computed indices
        gathered = torch.gather(x_flat, 1, gather_indices)  # Shape will be [B, h * w, C]
        out = gathered

        return out.view(B, N, C)

    return merge, unmerge


# def calculate_indices_vectorized(si, sj, h, w, new_h, new_w, x_device):
#     dtype = torch.float32
#     device = torch.device(x_device)

#     # 定义网格
#     global_i, global_j = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing='ij')

#     # 定义窗口的开始和结束位置
#     windowarea_start_i = (torch.arange(new_h, device=device) * si)
#     windowarea_start_j = (torch.arange(new_w, device=device) * sj)
#     windowarea_end_i = (start_is + si).clamp(max=h)
#     windowarea_end_j = (start_js + sj).clamp(max=w)

#     # 
#     # 初始化输出张量
#     merge_indices = torch.full((new_h, new_w, 2), -1, dtype=torch.long, device=device)
#     unmerge_indices = torch.full((h, w, 2), -1, dtype=torch.long, device=device)
#     max_area_coverage = torch.zeros((h, w), device=device)

#     # 对每个新窗口计算覆盖和索引
#     for i in range(new_h):
#         for j in range(new_w):
#             istart = start_is[i].floor()
#             jstart = start_js[j].floor()
#             iend = end_is[i].ceil()
#             jend = end_js[j].ceil()

#             local_is = global_i[istart:iend, jstart:jend]
#             local_js = global_j[istart:iend, jstart:jend]
#             area = (torch.min(local_is + 1, end_is[i]) - torch.max(local_is, start_is[i])) * \
#                    (torch.min(local_js + 1, end_js[j]) - torch.max(local_js, start_js[j]))
#             area = area.clamp(min=0)

#             max_area, max_pos = torch.max(area.view(-1), dim=0)
#             max_ii = max_pos // area.shape[1] + istart
#             max_jj = max_pos % area.shape[1] + jstart

#             merge_indices[i, j, :] = torch.tensor([max_ii, max_jj], dtype=torch.long, device=device)

#             mask = area > max_area_coverage[istart:iend, jstart:jend]
#             max_area_coverage[istart:iend, jstart:jend] = torch.where(mask, area, max_area_coverage[istart:iend, jstart:jend])
#             unmerge_indices[istart:iend, jstart:jend, 0][mask] = i
#             unmerge_indices[istart:iend, jstart:jend, 1][mask] = j

#     return merge_indices, unmerge_indices, max_area_coverage
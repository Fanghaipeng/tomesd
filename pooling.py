import torch
import torchvision.ops as ops

# 假设我们有一个特征图 `feature_map` 和一些感兴趣区域 `rois`
feature_map = torch.randn(1, 512, 32, 32)  # 一个简单的特征图，大小为 (N, C, H, W)
rois = torch.tensor([[0, 0, 0, 32, 32]]).float()  # 一个 ROI, 格式为 [batch_idx, x1, y1, x2, y2]

# 使用 ROI Pooling
output_size = (7, 9)  # 输出特征图的大小
pooled_features = ops.roi_pool(feature_map, rois, output_size)

print(pooled_features.shape)  # 输出应该是 (1, 512, 7, 7)
import torch

# fpn_feats is your input tensor
fpn_feats = torch.randn(3, 5)

# topk is your indices tensor
topk = torch.topk(fpn_feats, k=2, dim=1).indices

# Extract features from topk indices
features = fpn_feats[torch.arange(fpn_feats.size(0)).unsqueeze(1), topk]

print(features.size()) # Output: torch.Size([3, 2, 5])
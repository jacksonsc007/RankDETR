# test the validity
import torch

h = 10
w = 10

box_feature_range = torch.tensor([
    [
    [1, 1, 5, 5],
    [3, 3, 5, 8]],
    [
    [6, 7, 8, 9],
    [3, 8, 6, 9]]
]
    
)
bs, topk, _ = box_feature_range.size()
# Generate grid coordinates
grid_y, grid_x = torch.meshgrid(torch.arange(h), torch.arange(w))

# Expand dimensions to match the shape of box_feature_range
grid_y = grid_y.view(1, h, w, 1).to(device=box_feature_range.device)
grid_x = grid_x.view(1, h, w, 1).to(device=box_feature_range.device)


tmp =   (grid_x >= box_feature_range[..., 0].view(bs, 1, 1, topk)) & \
        (grid_x <  box_feature_range[..., 2].view(bs, 1, 1, topk)) & \
        (grid_y >= box_feature_range[..., 1].view(bs, 1, 1, topk)) & \
        (grid_y <  box_feature_range[..., 3].view(bs, 1, 1, topk))

extra_mask0 = ~(tmp.any(dim=-1))
print(extra_mask0)

extra_mask = extra_mask0.new_ones(bs, h, w, dtype=torch.bool)
extra_mask1 = extra_mask.clone()
extra_mask2 = extra_mask.clone()
for img_idx in range(bs):
    for loc_id in range(topk):
        lx, ly, rx, ry = box_feature_range[img_idx, loc_id]
        extra_mask1[img_idx][ly:ry, lx:rx] = False
        (extra_mask2[img_idx])[ly:ry, lx:rx] = False
print(extra_mask1)
print(extra_mask2)
print(torch.equal(extra_mask0, extra_mask1))
print(torch.equal(extra_mask0, extra_mask2))
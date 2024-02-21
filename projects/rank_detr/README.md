# version id: v3.0
In this version, we want to explore whether the cross-attention map could generate the reference points for deformable attention for encoder.

First we generate the cross-attention map with shape (bs, num_all_lvl_tokens, num_queries). The original encoder reference points have shape (bs, num_all_lvl_tokens, num_levels, 2), which will be expanded to (bs, num_all_lvl_toekns, num_heads, num_levels, num_points, 2) in the calculation of sampling_locations. It's obvious that the encoder reference points are same among different feature levels, and different final sampling locations. We want to improve this naive setting.

Experiments are based on [coco-minitrain](https://github.com/giddyyupp/coco-minitrain).


## version v3.0.1
In this minor version, we select top-k (k=num_levels) queries (including both one2one and one2many quereis) according to their class scores, whose corresponding box predictions are served as the reference points among all feature levels for each feature token. Indeed, this is a naive design. We just want to have a try.

Potential problems exist:
1. one-to-many queries tend to generate duplicate predictions, resulting in reference points belonging to same objects.
2. decoder predictions at early stages are not reliable.

### todo
How to improve:
1. only involve one2one queries.

2. For each token in encoder, we want it to focus on <num_points> objects. So we could use topk stardard to get <num_points> prediction boxes centers and its respective projection on each feature levels.

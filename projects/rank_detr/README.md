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

## version v3.0.2
In this minor version, each tokens will focus on <num_points> objects, whose centers server as reference points and are initialized according to the decoder cross-attention map. The locations of each objects centers are projected to feature maps of all levels.


### todo
1. ⭐
(priority-5, requires a lot of exps)
With the instruction of decoder cross-attention map, deformable attention may equip with more keypoints and reach a higher envelop. As the references points in the original deformable detr work are locations of tokens themselves. 

2. ⭐⭐⭐
Disentangle one-to-one queries as only one-to-many queries are discarded during inference. 

3. ⭐⭐⭐
Keep the locations of tokens as one of the reference points. This setting could avoid unreliable cross-attention map in the early stages ideally. 

4. ⭐⭐
In the original Deformable DETR, the reference points of deformable attention in encoder **are same among all encoder layers**, while the reference points in decoder are refined  progressively.
What if we use the sampling locations (reference_points + offset) of previous encoder layers as the references points of current layer? 

5. ⭐⭐⭐
reference points instructed by cross-attention map results in that  tokens have unstable regions to focus. This may cause performance drop. **Could we combine the cross-attention map of current decoder layer and all pervious ones??**

6. ⭐⭐
what if we retain the grad of object centers predicted by decoder and use them as the reference points in the deformable attention process in encoder?

## version v3.0.3
we only use one-to-one queries.

**results show that this won't help.**

## version v3.0.4
The locations of tokens themselves serve as one of the reference points. Both one2one and one2many queries are used.
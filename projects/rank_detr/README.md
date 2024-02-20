# version id: v1.1

In this version:
1. we use topk ratio to select the predictions with topk ratio classification score, including both one-to-one and one-to-many queries. 

2. each feature lvl is treated equivalently.

TODO:
As only one-to-one queries are used in inference, we consider using only prediction from one-to-one queries to generate mask. Besides, one-to-many queries tend to generate overlapping predictions with similar scores.

# version id: v1.2
1.using only prediction from one-to-one queries to generate mask.
2.add detach operation during mask generation. 


# version id: v1.3
Instead of mask both queries and keys, we consider reducing the query number following sparsedetr, aiming to improve precision and efficiency.

The basic idea is as sparse-detr, where only part of tokens(locations in the feature maps) are selected as queries, while the whole feature maps are treated as keys. 

Besides, the selected tokens will supplant the original ones in the src( which could be the original backbone feature or encoder hidden memory), while those unselected ones remain intact. (In a nutshell, only update selected tokens). This results in an output same size as the src.  

There are three minor versions which differ in the src and value: (check the src & encoder_value arg in func cascade_stage)

## v1.3.1
 Use cnn backbone feature as both src and value. 

## v1.3.2
 Use encoder hidden state (memory) as src, backbone feature as value.

## v1.3.3
 Use encoder hidden state (memory) as both src and value

Experiment results shows that the second one achieves better performance.

#TODO Selected keys instead of queries.


# version id: v2.0
In this major version, the cross attention map is used to select salient multi-lvl feature locations, following sparse-detr. For comparison, version 1.x make use of predicted boxes as the criterion for selection.

The basic idea is as follows:
1. determine the valid tokens numbers for each images after padding.
2. set a predefiend topk_ratio and multiply it with the valid tokens number for each image. The result serves as the selected token numbers for each image. Then select those tokens having topk cross-attention weights with object queries.


In this naive topk scheme, certain implementations may bring some drawbacks:
1. most queries are background queries, whose contribution to the calculation of cross-attention weights for multi-lvl features tokens should be weakened.
2. low-lvl feature tokens outnumber higher-lvls substantially. Feed multi-lvl tokens together to the topk selection process could be unfair for high-lvl tokens.
3. queries used to calculate attention weight (function: attn_map_to_flat_grid) include one-to-many queries, which will not be used in inference.

## v2.0.1
In this minor version, only one-to-one queries participate the calculation of function attn_map_to_flat_grid. The reason for making this minor change lies in that only one-to-one queries are used in inference.

Results showed that this modification **did not** help.

## v2.0.2
In this minor version, we want to get rid of the impact of background queries on calculation of the function attn_map_to_flat_grid. So:

1. first select object queries with topk classificiation score, which hopefully get rid of object queries corresponds to background.
2. only those selected object queries contribute to the calculation of the function attn_map_to_flat_grid.

Results showed that this modification **did not** help.

#TODO In later versions, we consider:
1. using the cross-attention map to initialize the reference points for encoder deformable attention.
2. fixed class threhold to select topk boxes during inference & training.

## v2.0.3
In this version, we start from v2.0, which means we use both one-to-one and one-to-many queries. There is only one change we want to make:

1. always keep the tokens from last feature maps

we want to validate whether a simple topk scheme among all feature levels will be detrimental to performance. The last feature level is outnumbered largely but the first feature level.

Results showed that this modification **did not** help.

## v2.0.4

In this version, we start from v2.0, which means we use both one-to-one and one-to-many queries. There is only one change we want to make:

1. we sample tokens from each feature level separately, which means we don't mix tokens from differents levels together when we're selecting according to the topk_ratio. 
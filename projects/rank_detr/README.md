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

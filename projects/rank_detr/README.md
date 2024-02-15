version id: v1.1

In this version:
1. we use topk ratio to select the predictions with topk ratio classification score, including both one-to-one and one-to-many queries. 

2. each feature lvl is treated equivalently.

TODO:
As only one-to-one queries are used in inference, we consider using only prediction from one-to-one queries to generate mask. Besides, one-to-many queries tend to generate overlapping predictions with similar scores.

version id: v1.2
1.using only prediction from one-to-one queries to generate mask.
2.add detach operation during mask generation. 
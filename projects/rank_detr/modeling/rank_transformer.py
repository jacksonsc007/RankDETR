# coding=utf-8
# Copyright 2022 The IDEA Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import math
import torch
import torch.nn as nn

from detrex.layers import (
    FFN,
    BaseTransformerLayer,
    MultiheadAttention,
    MultiScaleDeformableAttention,
    TransformerLayerSequence,
    box_cxcywh_to_xyxy
)
from detrex.utils import inverse_sigmoid

from detectron2.utils.events import get_event_storage
from fairscale.nn.checkpoint import checkpoint_wrapper

INK_INF = 1e20

class RankDetrTransformerEncoder(TransformerLayerSequence):
    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        feedforward_dim: int = 1024,
        attn_dropout: float = 0.1,
        ffn_dropout: float = 0.1,
        num_layers: int = 6,
        post_norm: bool = False,
        num_feature_levels: int = 4,
        use_checkpoint: bool = True,
    ):
        super(RankDetrTransformerEncoder, self).__init__(
            transformer_layers=BaseTransformerLayer(
                attn=MultiScaleDeformableAttention(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    dropout=attn_dropout,
                    batch_first=True,
                    num_levels=num_feature_levels,
                ),
                ffn=FFN(
                    embed_dim=embed_dim,
                    feedforward_dim=feedforward_dim,
                    output_dim=embed_dim,
                    num_fcs=2,
                    ffn_drop=ffn_dropout,
                ),
                norm=nn.LayerNorm(embed_dim),
                operation_order=("cross_attn", "norm", "ffn", "norm"),
            ),
            num_layers=num_layers,
        )
        self.embed_dim = self.layers[0].embed_dim
        self.pre_norm = self.layers[0].pre_norm

        if post_norm:
            self.post_norm_layer = nn.LayerNorm(self.embed_dim)
        else:
            self.post_norm_layer = None

        # use encoder checkpoint
        if use_checkpoint:
            for layer in self.layers:
                layer = checkpoint_wrapper(layer)

    def forward(
        self,
        query,
        key,
        value,
        query_pos=None,
        key_pos=None,
        attn_masks=None,
        query_key_padding_mask=None,
        key_padding_mask=None,
        **kwargs,
    ):

        for layer in self.layers:
            query = layer(
                query,
                key,
                value,
                query_pos=query_pos,
                attn_masks=attn_masks,
                query_key_padding_mask=query_key_padding_mask,
                key_padding_mask=key_padding_mask,
                **kwargs,
            )

        if self.post_norm_layer is not None:
            query = self.post_norm_layer(query)
        return query


class RankDetrTransformerDecoder(TransformerLayerSequence):
    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        feedforward_dim: int = 1024,
        attn_dropout: float = 0.1,
        ffn_dropout: float = 0.1,
        num_layers: int = 6,
        return_intermediate: bool = True,
        num_feature_levels: int = 4,
        use_checkpoint: bool = True,
        look_forward_twice=True,
        num_queries_one2one=300,
        num_queries_one2many=1500,
        two_stage_num_proposals=300,
        rank_adaptive_classhead=True,
        query_rank_layer=True,
    ):
        super(RankDetrTransformerDecoder, self).__init__(
            transformer_layers=BaseTransformerLayer(
                attn=[
                    MultiheadAttention(
                        embed_dim=embed_dim,
                        num_heads=num_heads,
                        attn_drop=attn_dropout,
                        batch_first=True,
                    ),
                    MultiScaleDeformableAttention(
                        embed_dim=embed_dim,
                        num_heads=num_heads,
                        dropout=attn_dropout,
                        batch_first=True,
                        num_levels=num_feature_levels,
                    ),
                ],
                ffn=FFN(
                    embed_dim=embed_dim,
                    feedforward_dim=feedforward_dim,
                    output_dim=embed_dim,
                    ffn_drop=ffn_dropout,
                ),
                norm=nn.LayerNorm(embed_dim),
                operation_order=(
                    "self_attn",
                    "norm",
                    "cross_attn",
                    "norm",
                    "ffn",
                    "norm",
                ),
            ),
            num_layers=num_layers,
        )
        self.return_intermediate = return_intermediate

        self.bbox_embed = None
        self.class_embed = None
        self.look_forward_twice = look_forward_twice

        # Rank-adaptive Classification Head
        self.rank_adaptive_classhead = rank_adaptive_classhead

        # query rank layer
        self.query_rank_layer = query_rank_layer
        self.num_queries_one2one = num_queries_one2one
        self.num_queries_one2many = num_queries_one2many
        if self.query_rank_layer:
            self.rank_aware_content_query = nn.ModuleList([
                copy.deepcopy(nn.Embedding(two_stage_num_proposals, embed_dim))
                for _ in range(num_layers - 1)
            ])
            for m in self.rank_aware_content_query.parameters():
                nn.init.zeros_(m)

            self.pre_racq_trans = nn.ModuleList([
                copy.deepcopy(nn.Linear(embed_dim, embed_dim))
                for _ in range(num_layers - 1)
            ])
            self.post_racq_trans = nn.ModuleList([
                copy.deepcopy(nn.Linear(embed_dim * 2, embed_dim))
                for _ in range(num_layers - 1)
            ])

        # decoder checkpoint
        if use_checkpoint:
            for layer in self.layers:
                layer = checkpoint_wrapper(layer)

    def forward(
        self,
        query,
        key,
        value,
        query_pos=None,
        key_pos=None,
        attn_masks=None,
        query_key_padding_mask=None,
        key_padding_mask=None,
        reference_points=None,
        valid_ratios=None,
        **kwargs,
    ):
        output = query

        intermediate = []
        intermediate_reference_points = []
        for layer_idx, layer in enumerate(self.layers):

            # query rank layer
            if layer_idx >= 1:
                if self.query_rank_layer:
                    output = torch.gather( # rank-aware content query
                        output, 1, rank_indices.unsqueeze(-1).repeat(1, 1, output.shape[-1])
                    )
                    concat_term = self.pre_racq_trans[layer_idx - 1](
                        self.rank_aware_content_query[layer_idx - 1].weight[:output.shape[1]].unsqueeze(0).expand(output.shape[0], -1, -1)
                    )
                    output = torch.cat((output, concat_term), dim=2)
                    output = self.post_racq_trans[layer_idx - 1](output)
                    query_pos = torch.gather( # rank-aware pos query
                        query_pos, 1, rank_indices.unsqueeze(-1).repeat(1, 1, query_pos.shape[-1])
                    )
                if (not self.query_rank_layer) and (self.rank_adaptive_classhead):
                    output = torch.gather(
                        output, 1, rank_indices.unsqueeze(-1).repeat(1, 1, output.shape[-1])
                    )
                    query_pos = torch.gather(
                        query_pos, 1, rank_indices.unsqueeze(-1).repeat(1, 1, query_pos.shape[-1])
                    )

            if reference_points.shape[-1] == 4:
                reference_points_input = ( # one-to-one queries + one-to-many queries == 1800
                    reference_points[:, :, None] # (bs, 1800, 1, 4) * (bs, 1, num_lvl, 4)
                    * torch.cat([valid_ratios, valid_ratios], -1)[:, None] # (bs, 1800, num_lvl, 4)
                )
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = reference_points[:, :, None] * valid_ratios[:, None]

            output = layer(
                output,
                key,
                value,
                query_pos=query_pos,
                key_pos=key_pos,
                attn_masks=attn_masks,
                query_key_padding_mask=query_key_padding_mask,
                key_padding_mask=key_padding_mask,
                reference_points=reference_points_input,
                **kwargs,
            )

            if self.bbox_embed is not None:
                tmp = self.bbox_embed[layer_idx](output)
                if reference_points.shape[-1] == 4:
                    new_reference_points = tmp + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                else:
                    assert reference_points.shape[-1] == 2
                    new_reference_points = tmp
                    new_reference_points[..., :2] = tmp[..., :2] + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach() # NOTE gradient detached

            if self.return_intermediate:

                if (layer_idx >= 0) and (self.query_rank_layer or self.rank_adaptive_classhead):
                    # generate rank indices
                    outputs_class_tmp = self.class_embed[layer_idx](output)  # [bs, num_queries, embed_dim] -> [bs, num_queries, num_classes]
                    rank_basis = outputs_class_tmp.sigmoid().max(dim=2, keepdim=False)[0] # tensor shape: [bs, num_queries]
                    if self.training:
                        rank_indices_one2one  = torch.argsort(rank_basis[:, : self.num_queries_one2one], dim=1, descending=True) # tensor shape: [bs, num_queries_one2one]
                        rank_indices_one2many = torch.argsort(rank_basis[:, self.num_queries_one2one :], dim=1, descending=True) # tensor shape: [bs, num_queries_one2many]
                        rank_indices = torch.cat(
                            (
                                rank_indices_one2one,
                                rank_indices_one2many + torch.ones_like(rank_indices_one2many) * self.num_queries_one2one
                            ),
                            dim=1,
                        ) # tensor shape: [bs, num_queries_one2one+num_queries_one2many]
                    else:
                        rank_indices = torch.argsort(rank_basis[:, : self.num_queries_one2one], dim=1, descending=True)
                    rank_indices = rank_indices.detach() # NOTE detach
                    # rank the reference points
                    reference_points = torch.gather( # (bs, num_queries, 1) -> (bs, num_queries, 4)
                        reference_points, 1, rank_indices.unsqueeze(-1).repeat(1, 1, reference_points.shape[-1]))
                    new_reference_points = torch.gather(
                        new_reference_points, 1, rank_indices.unsqueeze(-1).repeat(1, 1, new_reference_points.shape[-1]))

                intermediate.append(output)
                intermediate_reference_points.append( # reference_points now is the gradient detached form of new_reference points.
                    new_reference_points if self.look_forward_twice else reference_points # TODO look forward twice? Refer to DINO sec3.5.
                )

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_reference_points)

        return output, reference_points


class RankDetrTransformer(nn.Module):
    """Transformer module for Deformable DETR

    Args:
        encoder (nn.Module): encoder module.
        decoder (nn.Module): decoder module.
        as_two_stage (bool): whether to use two-stage transformer. Default False.
        num_feature_levels (int): number of feature levels. Default 4.
        two_stage_num_proposals (int): number of proposals in two-stage transformer. Default 300.
            Only used when as_two_stage is True.
    """

    def __init__(
        self,
        encoder=None,
        decoder=None,
        num_feature_levels=4,
        as_two_stage=False,
        num_queries_one2one=300,
        num_queries_one2many=1500,
        two_stage_num_proposals=300,
        mixed_selection=True,
        rank_adaptive_classhead=True,
        topk_ratio=0.5
    ):
        super(RankDetrTransformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.num_feature_levels = num_feature_levels
        self.as_two_stage = as_two_stage
        self.two_stage_num_proposals = two_stage_num_proposals

        self.embed_dim = self.encoder.embed_dim

        self.level_embeds = nn.Parameter(torch.Tensor(self.num_feature_levels, self.embed_dim))

        if self.as_two_stage:
            self.enc_output = nn.Linear(self.embed_dim, self.embed_dim)
            self.enc_output_norm = nn.LayerNorm(self.embed_dim)
            self.pos_trans = nn.Linear(self.embed_dim * 2, self.embed_dim * 2)
            self.pos_trans_norm = nn.LayerNorm(self.embed_dim * 2)
        else:
            self.reference_points_trans = nn.Linear(self.embed_dim, 2)

        self.mixed_selection = mixed_selection

        self.init_weights()

        assert self.encoder.num_layers == self.decoder.num_layers, \
        "symmetric encoder decoders design is now required."
        self.num_stages = self.encoder.num_layers

        self.topk_ratio=topk_ratio

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MultiScaleDeformableAttention):
                m.init_weights()
        if not self.as_two_stage:
            nn.init.xavier_normal_(self.reference_points.weight.data, gain=1.0)
            nn.init.constant_(self.reference_points.bias.data, 0.0)
        nn.init.normal_(self.level_embeds)

    def gen_encoder_output_proposals(self, memory, memory_padding_mask, spatial_shapes):
        N, S, C = memory.shape # batchsize, all_lvl_loc_num, feat_channel
        proposals = []
        _cur = 0                      

        for lvl, (H, W) in enumerate(spatial_shapes):
            mask_flatten_ = memory_padding_mask[:, _cur : (_cur + H * W)].view(N, H, W, 1)
            valid_H = torch.sum(~mask_flatten_[:, :, 0, 0], 1) # (N, H) -> (N, )
            valid_W = torch.sum(~mask_flatten_[:, 0, :, 0], 1) # (N, W) -> (N, )

            grid_y, grid_x = torch.meshgrid( # (H, W)
                torch.linspace(0, H - 1, H, dtype=torch.float32, device=memory.device),
                torch.linspace(0, W - 1, W, dtype=torch.float32, device=memory.device),
            )
            grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1) # 2*(H, W) -> (H,W,2)

            scale = torch.cat([valid_W.unsqueeze(-1), valid_H.unsqueeze(-1)], 1).view(N, 1, 1, 2) # (N, 1, 1, 2)
            grid = (grid.unsqueeze(0).expand(N, -1, -1, -1) + 0.5) / scale # (1, H, W, 2) -> (N, H, W, 2) 
            wh = torch.ones_like(grid) * 0.05 * (2.0**lvl) # Appendix A.4 in deformable-detr
            proposal = torch.cat((grid, wh), -1).view(N, -1, 4) # proposal representation: normalized (c_x, c_y, w, h)
            proposals.append(proposal)
            _cur += H * W

        output_proposals = torch.cat(proposals, 1) # (bs, all_lvl_loc_num,4)
        output_proposals_valid = ((output_proposals > 0.01) & (output_proposals < 0.99)).all(
            -1, keepdim=True # (bs, all_lvl_loc_num,1)
        )
        output_proposals = torch.log(output_proposals / (1 - output_proposals)) # inverse sigmoid
        output_proposals = output_proposals.masked_fill(
            memory_padding_mask.unsqueeze(-1), float("inf")
        )
        output_proposals = output_proposals.masked_fill(~output_proposals_valid, float("inf"))

        output_memory = memory
        output_memory = output_memory.masked_fill(memory_padding_mask.unsqueeze(-1), float(0))
        output_memory = output_memory.masked_fill(~output_proposals_valid, float(0))
        output_memory = self.enc_output_norm(self.enc_output(output_memory))
        return output_memory, output_proposals

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        """Get the reference points used in decoder.

        Args:
            spatial_shapes (Tensor): The shape of all
                feature maps, has shape (num_level, 2).
            valid_ratios (Tensor): The radios of valid
                points on the feature map, has shape
                (bs, num_levels, 2)
            device (obj:`device`): The device where
                reference_points should be.

        Returns:
            Tensor: reference points used in decoder, has \
                shape (bs, num_keys, num_levels, 2).
        """
        reference_points_list = []
        for lvl, (H, W) in enumerate(spatial_shapes):
            #  TODO  check this 0.5
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(0.5, H - 0.5, H, dtype=torch.float32, device=device),
                torch.linspace(0.5, W - 0.5, W, dtype=torch.float32, device=device),
            )
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H) # (1, h*w) / (bs, 1) -> (bs, h*w)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W)
            ref = torch.stack((ref_x, ref_y), -1) # (bs, h*w, 2)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None] # (bs,all_lvl_num, 1, 2) * (bs, 1, num_lvl, 2)
        return reference_points

    def get_valid_ratio(self, mask):
        """Get the valid radios of feature maps of all  level."""
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def get_proposal_pos_embed(self, proposals, num_pos_feats=128, temperature=10000):
        """Get the position embedding of proposal."""
        scale = 2 * math.pi
        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=proposals.device)
        dim_t = temperature ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / num_pos_feats)
        # N, L, 4
        proposals = proposals.sigmoid() * scale
        # N, L, 4, 128
        pos = proposals[:, :, :, None] / dim_t
        # N, L, 4, 64, 2 -> N, L, 512
        pos = torch.stack((pos[:, :, :, 0::2].sin(), pos[:, :, :, 1::2].cos()), dim=4).flatten(2)
        return pos

    def forward(
        self,
        multi_level_feats,
        multi_level_masks,
        multi_level_pos_embeds,
        query_embed,
        self_attn_mask,
        img_true_sizes,
        img_batched_sizes,
        **kwargs,
    ):
        assert self.as_two_stage or query_embed is not None

        feat_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        multi_lvl_feature_valid_size = []
        for lvl, (feat, mask, pos_embed) in enumerate(
            zip(multi_level_feats, multi_level_masks, multi_level_pos_embeds)
        ):
            bs, c, h, w = feat.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            valid_h = (~mask).cumsum(dim=1)[:, -1, 0]
            valid_w = (~mask).cumsum(dim=2)[:, 0, -1]
            valid_size = torch.stack([valid_h, valid_w], dim = -1)
            multi_lvl_feature_valid_size.append(valid_size)

            feat = feat.flatten(2).transpose(1, 2)  # bs, hw, c
            mask = mask.flatten(1) # bs, hw
            pos_embed = pos_embed.flatten(2).transpose(1, 2)  # bs, hw, c
            lvl_pos_embed = pos_embed + self.level_embeds[lvl].view(1, 1, -1) # level_embed to distinguish levels?
            lvl_pos_embed_flatten.append(lvl_pos_embed)



            feat_flatten.append(feat)
            mask_flatten.append(mask)
        feat_flatten = torch.cat(feat_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=feat_flatten.device # featuer map shape
        )
        level_start_index = torch.cat(
            (spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1])
        )
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in multi_level_masks], 1) # images have invalid feature locations in feature maps, due to padding for batching.

        encoder_reference_points = self.get_reference_points( # TODO why twice ratio?
            spatial_shapes, valid_ratios, device=feat.device
        )


        decoder_query = None
        decoder_query_pos = None
        decoder_reference_points = None
        rank_indices = None

        inter_states = []
        inter_references = []
        init_reference_outs = []
        enc_outputs_class_all = []
        enc_outputs_coord_unact_all = []
        # topk = int(self.num_queries * self.topk_ratio)
        
        sparse_enc_query = feat_flatten
        sparse_enc_query_pos = lvl_pos_embed_flatten
        sparse_enc_ref = encoder_reference_points
        topk_enc_token_indice = None
        valid_enc_token_num = None
        # src = feat_flatten
        memory = feat_flatten
        for stage_id in range(self.num_stages):
            memory, decoder_query, decoder_query_pos,\
            rank_indices, decoder_reference_points, new_reference_points, decoder_output_classes, init_reference_out, \
            enc_outputs_class, enc_outputs_coord_unact = \
                self.cascade_stage(
                    stage_id=stage_id,
                    src=memory,
                    encoder_query=sparse_enc_query,
                    encoder_key=None, # key is not required in deformable attention
                    encoder_value=memory,
                    encoder_query_pos=sparse_enc_query_pos,
                    encoder_attn_masks = None,
                    encoder_reference_points=sparse_enc_ref,
                    topk_enc_token_indice=topk_enc_token_indice,
                    valid_enc_token_num=valid_enc_token_num,
                    query_key_padding_mask=mask_flatten,
                    decoder_query=decoder_query,
                    decoder_query_pos=decoder_query_pos,
                    decoder_query_embed=query_embed,
                    decoder_reference_points=decoder_reference_points,
                    decoder_attn_masks = [self_attn_mask, None],
                    spatial_shapes=spatial_shapes,
                    level_start_index=level_start_index,
                    valid_ratios=valid_ratios,
                    rank_indices=rank_indices,
                    **kwargs
            )
            assert memory.isfinite().all()
            
            if stage_id != self.num_stages:
                extra_masks = self.mask_from_decoder( decoder_reference_points, decoder_output_classes, img_true_sizes, \
                                                img_batched_sizes, multi_level_feats, self.topk_ratio, multi_lvl_feature_valid_size)

                assert extra_masks.size() == mask_flatten.size()
                new_query_key_padding_mask = ( extra_masks | mask_flatten )
            
                # wandb logging
                # new_query_key_padding_mask = new_query_key_padding_mask.float().masked_fill(new_query_key_padding_mask, -INK_INF) 
                padding_mask_ratio = ((~mask_flatten).int().sum() / mask_flatten.numel()).item()
                decoder_mask_ratio = ((~extra_masks).int().sum() / extra_masks.numel()).item()
                new_mask_ratio = ((~new_query_key_padding_mask).int().sum() / new_query_key_padding_mask.numel()).item()
                storage = get_event_storage()
                storage.put_scalar('mask/padding_mask_ratio', padding_mask_ratio)
                storage.put_scalar('mask/decoder_mask_ratio', decoder_mask_ratio)
                storage.put_scalar('mask/new_mask_ratio', new_mask_ratio)

                valid_enc_token_num = (~new_query_key_padding_mask).int().sum(dim=1)
                batch_token_num = int(max(valid_enc_token_num))
                topk_enc_token_indice = (~new_query_key_padding_mask).int().topk(batch_token_num, dim=1)[1] # (bs, batch_token_num)
                feature_dim = memory.size(2)
                sparse_enc_query = memory.gather(dim=1, index=topk_enc_token_indice.unsqueeze(dim=2).repeat(1, 1, feature_dim))
                sparse_enc_query_pos = lvl_pos_embed_flatten.gather(dim=1, index=topk_enc_token_indice.unsqueeze(dim=2).repeat(1, 1, feature_dim))
                num_lvl = encoder_reference_points.size(2)
                sparse_enc_ref = encoder_reference_points.gather(dim=1, index=topk_enc_token_indice.unsqueeze(dim=2).unsqueeze(dim=3).repeat(1, 1, num_lvl, 2)) # (x, y) for ref points
            



            # assert decoder_reference_points.requires_grad == False
            inter_states.append(decoder_query)
            inter_references.append( new_reference_points if self.decoder.look_forward_twice else decoder_reference_points)
            init_reference_outs.append(init_reference_out)
            enc_outputs_class_all.append(enc_outputs_class)
            enc_outputs_coord_unact_all.append(enc_outputs_coord_unact)

        if self.decoder.return_intermediate:
            inter_states = torch.stack(inter_states)
            inter_references = torch.stack(inter_references)
        else:
            inter_states = decoder_query
            inter_references = decoder_reference_points
        inter_references_out = inter_references
        if self.as_two_stage:
            return (
                inter_states,
                init_reference_outs[0],
                inter_references_out,
                enc_outputs_class_all[0],
                enc_outputs_coord_unact_all[0],
            )
        return inter_states, init_reference_outs[0], inter_references_out, None, None
    
    def mask_from_decoder(self, output_coord, output_class, img_true_size, img_batched_sizes, multi_level_feats, topk_ratio, multi_lvl_feature_valid_size):
        # topk=10
        bs, num_queries, _= output_coord.shape
        topk = int(self.num_queries_one2one * topk_ratio )
        # assert topk < num_queries
        # batched_h, batch_w = img_batched_sizes
        
        # use one2one queries 
        output_class = output_class[:, :self.num_queries_one2one ].detach()
        output_coord = output_coord[:, :self.num_queries_one2one ].detach()

        output_class = output_class.max(dim=-1)[0] # (bs, num_query)
        topkid = output_class.topk(topk, dim= -1)[1] # (bs, topk)
        topk_boxes = torch.gather(output_coord, 1, topkid[..., None].repeat(1, 1, 4)) # (bs, topk, 4)
        topk_boxes = box_cxcywh_to_xyxy(topk_boxes).clamp(min=0, max=1)
        extra_masks = []
        for lvl, (feat, valid_feature_size) in enumerate(
                zip(multi_level_feats, multi_lvl_feature_valid_size)
            ):
            bs, _, h, w = feat.shape
            valid_feature_size = valid_feature_size.repeat(1, 2).view(bs, 1, 4)

            # we omit the difference between true size and batched size here
            box_feature_range = ( topk_boxes * valid_feature_size ).floor().to(torch.int64)
            assert  (box_feature_range[..., :2] <= box_feature_range[..., 2:]).all()

            # extra_mask = output_coord.new_ones(bs, h, w, dtype=torch.bool) # 0 mean valid, 1 means invalid
            # extra_mask1 = extra_mask.clone()
            # extra_mask2 = extra_mask.clone()
            # for img_idx in range(bs):
            #     for loc_id in range(topk):
            #         lx, ly, rx, ry = box_feature_range[img_idx, loc_id]
            #         extra_mask1[img_idx][ly:ry, lx:rx] = False
            #         (extra_mask2[img_idx])[ly:ry, lx:rx] = False
            

            # Generate grid coordinates
            grid_y, grid_x = torch.meshgrid(torch.arange(h), torch.arange(w))

            # Expand dimensions to match the shape of box_feature_range
            grid_y = grid_y.view(1, h, w, 1).to(device=box_feature_range.device)
            grid_x = grid_x.view(1, h, w, 1).to(device=box_feature_range.device)


            tmp =   (grid_x >= box_feature_range[..., 0].view(bs, 1, 1, topk)) & \
                    (grid_x <  box_feature_range[..., 2].view(bs, 1, 1, topk)) & \
                    (grid_y >= box_feature_range[..., 1].view(bs, 1, 1, topk)) & \
                    (grid_y <  box_feature_range[..., 3].view(bs, 1, 1, topk))
            assert tmp.shape == torch.Size([bs, h, w, topk])
            extra_mask = ~(tmp.any(dim=-1))
            extra_mask = extra_mask.view(bs, -1)
            extra_masks.append(extra_mask)
        extra_masks = torch.cat(extra_masks, dim=1)
       
        
        return extra_masks


    def cascade_stage(
        self,
        stage_id,
        src,
        encoder_query,
        encoder_key,
        encoder_value,
        encoder_query_pos,
        encoder_attn_masks,
        encoder_reference_points,
        topk_enc_token_indice,
        valid_enc_token_num,
        query_key_padding_mask,
        decoder_query,
        decoder_query_pos,
        decoder_query_embed,
        decoder_reference_points,
        decoder_attn_masks,
        spatial_shapes,
        level_start_index,
        valid_ratios,
        rank_indices,
        **kwargs
    ):
        memory = self.cascade_stage_encoder_part(
            stage_id,
            src=src,
            query=encoder_query,
            key=encoder_key,
            value=encoder_value,
            query_pos=encoder_query_pos,
            key_pos=None, # key pos is not required in deformable attention
            attn_masks=encoder_attn_masks,
            key_padding_mask=query_key_padding_mask, # encoder now adopt cross-attention
            spatial_shapes=spatial_shapes,
            reference_points=encoder_reference_points,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            topk_enc_token_indice=topk_enc_token_indice,
            valid_enc_token_num=valid_enc_token_num,
            **kwargs,
        )

        init_reference_out = None
        enc_outputs_class = None
        enc_outputs_coord_unact = None

        # generate initial query for decoder 
        if stage_id == 0: 
            bs, _, c = memory.shape
            if self.as_two_stage:
                output_memory, output_proposals = self.gen_encoder_output_proposals(
                    memory, query_key_padding_mask, spatial_shapes
                )

                enc_outputs_class = self.decoder.class_embed[self.decoder.num_layers](output_memory)
                enc_outputs_coord_unact = (
                    self.decoder.bbox_embed[self.decoder.num_layers](output_memory) + output_proposals
                )

                topk = self.two_stage_num_proposals
                topk_proposals = torch.topk(enc_outputs_class[..., 0], topk, dim=1)[1]
                topk_coords_unact = torch.gather(
                    enc_outputs_coord_unact, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 4)
                )
                topk_coords_unact = topk_coords_unact.detach()
                decoder_reference_points = topk_coords_unact.sigmoid()
                init_reference_out = decoder_reference_points
                pos_trans_out = self.pos_trans_norm(
                    self.pos_trans(self.get_proposal_pos_embed(topk_coords_unact))
                )
                if not self.mixed_selection:
                    decoder_query_pos, decoder_query = torch.split(pos_trans_out, c, dim=2)
                else:
                    # decoder_query_pos here is the content embed for deformable DETR
                    decoder_query = decoder_query_embed.unsqueeze(0).expand(bs, -1, -1)
                    decoder_query_pos, _ = torch.split(pos_trans_out, c, dim=2)
            else:
                decoder_query_pos, decoder_query = torch.split(decoder_query_embed, c, dim=1)
                decoder_query_pos = decoder_query_pos.unsqueeze(0).expand(bs, -1, -1)
                decoder_query = decoder_query.unsqueeze(0).expand(bs, -1, -1)
                decoder_reference_points = self.reference_points_trans(decoder_query_pos).sigmoid()
                init_reference_out = decoder_reference_points


        decoder_query, decoder_query_pos, rank_indices, \
            decoder_reference_points, new_reference_points, decoder_output_classes =  \
                self.cascade_stage_decoder_part(
                    stage_id,
                    query=decoder_query,  # bs, num_queries, embed_dims
                    key=None,  # bs, num_tokens, embed_dims
                    value=memory,  # bs, num_tokens, embed_dims
                    query_pos=decoder_query_pos,
                    key_padding_mask=query_key_padding_mask,  # bs, num_tokens
                    reference_points=decoder_reference_points,  # num_queries, 4
                    spatial_shapes=spatial_shapes,  # nlvl, 2
                    level_start_index=level_start_index,  # nlvl
                    valid_ratios=valid_ratios,  # bs, nlvl, 2
                    attn_masks=decoder_attn_masks,
                    rank_indices=rank_indices,
                    **kwargs,
        )

        


        return ( memory, decoder_query, decoder_query_pos, rank_indices, \
                 decoder_reference_points, new_reference_points, decoder_output_classes, init_reference_out,\
                 enc_outputs_class, enc_outputs_coord_unact)
    
        
    def cascade_stage_encoder_part(
        self,
        stage_id,
        src,
        query,
        key,
        value,
        query_pos=None,
        key_pos=None,
        attn_masks=None,
        query_key_padding_mask=None,
        key_padding_mask=None,
        topk_enc_token_indice=None,
        valid_enc_token_num=None,
        **kwargs,
    ):
        encoder_layer = self.encoder.layers[stage_id] 
        sparse_memory = encoder_layer(
            query,
            key,
            value,
            query_pos=query_pos,
            key_pos=key_pos, 
            attn_masks=attn_masks,
            query_key_padding_mask=query_key_padding_mask,
            key_padding_mask=key_padding_mask,
            **kwargs, 
        )

        if topk_enc_token_indice is not None:
            assert valid_enc_token_num is not None
            bs = topk_enc_token_indice.size(0) 
            outputs=[]
            for img_id, (num, idx) in enumerate(zip(valid_enc_token_num, topk_enc_token_indice)):
                valid_idx = idx[:num]
                # src[0]: (ori_num_token, 256)
                # idx: (valid_num,) -> (valid_num, 256)
                # sparse_memory[0]: (max_token_num, 256)
                # src[0][index[i,j], j] = sparse_memory[0][i,j]
                outputs.append(src[img_id].scatter(dim=0, index=valid_idx.unsqueeze(1).repeat(1, src.size(-1)), src=sparse_memory[img_id][:num]))
            memory = torch.stack(outputs)
            assert memory.size(0) == bs

        else: 
            memory=sparse_memory
                

            

        if stage_id == (self.num_stages - 1) and self.encoder.post_norm_layer is not None :
            memory = self.encoder.post_norm_layer(memory)
        return memory
       
        

    def cascade_stage_decoder_part(
        self,
        stage_id,
        query,
        key,
        value,
        query_pos=None,
        key_pos=None,
        attn_masks=None,
        query_key_padding_mask=None,
        key_padding_mask=None,
        reference_points=None,
        valid_ratios=None,
        rank_indices=None,
        **kwargs,
    ):
        output = query

        # query rank layer
        if stage_id >= 1:
            assert rank_indices is not None
            if self.decoder.query_rank_layer:
                output = torch.gather(
                    output, 1, rank_indices.unsqueeze(-1).repeat(1, 1, output.shape[-1])
                )
                concat_term = self.decoder.pre_racq_trans[stage_id - 1](
                    self.decoder.rank_aware_content_query[stage_id - 1].weight[:output.shape[1]].unsqueeze(0).expand(output.shape[0], -1, -1)
                )
                output = torch.cat((output, concat_term), dim=2)
                output = self.decoder.post_racq_trans[stage_id - 1](output)
                query_pos = torch.gather(
                    query_pos, 1, rank_indices.unsqueeze(-1).repeat(1, 1, query_pos.shape[-1])
                )
            if (not self.decoder.query_rank_layer) and (self.decoder.rank_adaptive_classhead):
                output = torch.gather(
                    output, 1, rank_indices.unsqueeze(-1).repeat(1, 1, output.shape[-1])
                )
                query_pos = torch.gather(
                    query_pos, 1, rank_indices.unsqueeze(-1).repeat(1, 1, query_pos.shape[-1])
                )

        if reference_points.shape[-1] == 4:
            reference_points_input = (
                reference_points[:, :, None]
                * torch.cat([valid_ratios, valid_ratios], -1)[:, None]
            )
        else:
            assert reference_points.shape[-1] == 2
            reference_points_input = reference_points[:, :, None] * valid_ratios[:, None] 

        layer = self.decoder.layers[stage_id]
        output = layer(
            output,
            key,
            value,
            query_pos=query_pos,
            key_pos=key_pos,
            attn_masks=attn_masks,
            query_key_padding_mask=query_key_padding_mask,
            key_padding_mask=key_padding_mask,
            reference_points=reference_points_input,
            **kwargs,
        )

        if self.decoder.bbox_embed is not None:
            tmp = self.decoder.bbox_embed[stage_id](output)
            if reference_points.shape[-1] == 4:
                new_reference_points = tmp + inverse_sigmoid(reference_points)
                new_reference_points = new_reference_points.sigmoid()
            else:
                assert reference_points.shape[-1] == 2
                new_reference_points = tmp
                new_reference_points[..., :2] = tmp[..., :2] + inverse_sigmoid(reference_points)
                new_reference_points = new_reference_points.sigmoid()
            reference_points = new_reference_points.detach()

        if self.decoder.return_intermediate:

            if (stage_id >= 0) and (self.decoder.query_rank_layer or self.decoder.rank_adaptive_classhead):
                # generate rank indices
                outputs_class_tmp = self.decoder.class_embed[stage_id](output).sigmoid()  # [bs, num_queries, embed_dim] -> [bs, num_queries, num_classes]
                rank_basis = outputs_class_tmp.max(dim=2, keepdim=False)[0] # tensor shape: [bs, num_queries]
                if self.decoder.training:
                    rank_indices_one2one  = torch.argsort(rank_basis[:, : self.decoder.num_queries_one2one], dim=1, descending=True) # tensor shape: [bs, num_queries_one2one]
                    rank_indices_one2many = torch.argsort(rank_basis[:, self.decoder.num_queries_one2one :], dim=1, descending=True) # tensor shape: [bs, num_queries_one2many]
                    rank_indices = torch.cat(
                        (
                            rank_indices_one2one,
                            rank_indices_one2many + torch.ones_like(rank_indices_one2many) * self.decoder.num_queries_one2one
                        ),
                        dim=1,
                    ) # tensor shape: [bs, num_queries_one2one+num_queries_one2many]
                else:
                    rank_indices = torch.argsort(rank_basis[:, : self.decoder.num_queries_one2one], dim=1, descending=True)
                rank_indices = rank_indices.detach()
                # rank the reference points
                reference_points = torch.gather(
                    reference_points, 1, rank_indices.unsqueeze(-1).repeat(1, 1, reference_points.shape[-1]))
                new_reference_points = torch.gather(
                    new_reference_points, 1, rank_indices.unsqueeze(-1).repeat(1, 1, new_reference_points.shape[-1]))
        
        
        return output, query_pos, rank_indices, reference_points, new_reference_points, outputs_class_tmp

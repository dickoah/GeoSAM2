"""The GeoSAM2 model, built in Python.

This replaces Hydra's ``instantiate`` over VAST's ``configs/geosam2.yaml``: the
same classes with the same arguments, in the same order, so the module tree
and the parameter names are what the checkpoint expects -- ``strict=True``
below is what guarantees it. Both builders return the model on the CPU, in
eval mode; the segmenter moves it to the GPU for a call.
"""

from __future__ import annotations

import logging

import torch

from geosam2.sam2.modeling.backbones.hieradet import Hiera
from geosam2.sam2.modeling.backbones.image_encoder import FpnNeck
from geosam2.sam2.modeling.feature_fusion import FeatureFusion
from geosam2.sam2.modeling.memory_attention import MemoryAttention, MemoryAttentionLayer
from geosam2.sam2.modeling.memory_encoder import CXBlock, Fuser, MaskDownSampler, MemoryEncoder
from geosam2.sam2.modeling.position_encoding import PositionEmbeddingSine
from geosam2.sam2.modeling.sam.lora import SAMLoraImgEncoder
from geosam2.sam2.modeling.sam.transformer import RoPEAttention
from geosam2.sam2.modeling.sam2_base_geosam2 import SAM2Base
from geosam2.sam2.sam2_video_predictor_geosam2 import SAM2VideoPredictor

# What VAST's build added as Hydra overrides for the video predictor.
_DECODER_POSTPROCESSING = dict(
    dynamic_multimask_via_stability=True,
    dynamic_multimask_stability_delta=0.05,
    dynamic_multimask_stability_thresh=0.98,
)


def _lora_image_encoder(drop_path_rate=None):
    """One of the two Hiera + FPN encoders (images, position maps), LoRA-wrapped."""
    trunk_args = dict(embed_dim=112, num_heads=2)
    if drop_path_rate is not None:
        trunk_args["drop_path_rate"] = drop_path_rate
    return SAMLoraImgEncoder(
        rank=4,
        scalp=1,
        trunk=Hiera(**trunk_args),
        neck=FpnNeck(
            position_encoding=PositionEmbeddingSine(
                num_pos_feats=256, normalize=True, scale=None, temperature=10000),
            d_model=256,
            backbone_channel_list=[896, 448, 224, 112],
            fpn_top_down_levels=[2, 3],
            fpn_interp_model="nearest",
        ),
    )


def _rope_attention(**extra):
    return RoPEAttention(rope_theta=10000.0, feat_sizes=[64, 64], embedding_dim=256,
                         num_heads=1, downsample_rate=1, dropout=0.1, **extra)


def _components():
    """The sub-modules, built in the YAML's order (it decides how the RNG is consumed)."""
    image_encoder = _lora_image_encoder()
    pos_map_encoder = _lora_image_encoder(drop_path_rate=0.1)
    memory_attention = MemoryAttention(
        d_model=256,
        pos_enc_at_input=True,
        layer=MemoryAttentionLayer(
            activation="relu",
            dim_feedforward=2048,
            dropout=0.1,
            pos_enc_at_attn=False,
            self_attention=_rope_attention(),
            d_model=256,
            pos_enc_at_cross_attn_keys=True,
            pos_enc_at_cross_attn_queries=False,
            cross_attention=_rope_attention(rope_k_repeat=True, kv_in_dim=64),
        ),
        num_layers=4,
    )
    memory_encoder = MemoryEncoder(
        out_dim=64,
        position_encoding=PositionEmbeddingSine(
            num_pos_feats=64, normalize=True, scale=None, temperature=10000),
        mask_downsampler=MaskDownSampler(kernel_size=3, stride=2, padding=1),
        fuser=Fuser(
            layer=CXBlock(dim=256, kernel_size=7, padding=3,
                          layer_scale_init_value=1e-6, use_dwconv=True),
            num_layers=2,
        ),
    )
    feature_fusion = FeatureFusion(in_channels=256, out_channels=256)
    return dict(
        image_encoder=image_encoder,
        pos_map_encoder=pos_map_encoder,
        memory_attention=memory_attention,
        memory_encoder=memory_encoder,
        feature_fusion=feature_fusion,
    )


_BASE_ARGS = dict(
    num_maskmem=7,
    image_size=1024,
    sigmoid_scale_for_mem_enc=20.0,
    sigmoid_bias_for_mem_enc=-10.0,
    use_mask_input_as_output_without_sam=True,
    directly_add_no_mem_embed=True,
    no_obj_embed_spatial=True,
    use_high_res_features_in_sam=True,
    multimask_output_in_sam=True,
    iou_prediction_use_sigmoid=True,
    use_obj_ptrs_in_encoder=True,
    add_tpos_enc_to_obj_ptrs=True,
    proj_tpos_enc_in_obj_ptrs=True,
    use_signed_tpos_enc_to_obj_ptrs=True,
    only_obj_ptrs_in_the_past_for_eval=True,
    pred_obj_scores=True,
    pred_obj_scores_mlp=True,
    fixed_no_obj_ptr=True,
    multimask_output_for_tracking=True,
    use_multimask_token_for_obj_ptr=True,
    multimask_min_pt_num=0,
    multimask_max_pt_num=1,
    use_mlp_for_obj_ptr_proj=True,
    compile_image_encoder=False,
)


def image_model(checkpoint: str) -> SAM2Base:
    """The image-level model the automatic mask generator drives (no decoder post-processing)."""
    return _load(SAM2Base(**_components(), **_BASE_ARGS), checkpoint)


def video_predictor(checkpoint: str) -> SAM2VideoPredictor:
    """The video predictor the multi-view propagation runs on."""
    model = SAM2VideoPredictor(
        **_components(), **_BASE_ARGS,
        sam_mask_decoder_extra_args=dict(_DECODER_POSTPROCESSING),
        binarize_mask_from_pts_for_mem_enc=True,
        fill_hole_area=8,
    )
    return _load(model, checkpoint)


def _load(model, checkpoint: str):
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)["model"]
    model.load_state_dict(state, strict=True)
    logging.info("Loaded checkpoint from %s", checkpoint)
    return model.eval()

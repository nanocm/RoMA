import math
from functools import partial
from typing import Mapping, Optional

import torch
import torch.nn as nn
from mamba_ssm.utils.generation import GenerationMixin
from mamba_ssm.utils.hf import load_config_hf, load_state_dict_hf

# from mmdet.utils import get_root_logger
import logging
def get_root_logger(log_file=None, log_level=logging.INFO):
    """Get root logger.

    Args:
        log_file (str, optional): File path of log. Defaults to None.
        log_level (int, optional): The level of logger.
            Defaults to logging.INFO.

    Returns:
        :obj:`logging.Logger`: The obtained logger
    """
    logger = get_logger(name='mmdet', log_file=log_file, log_level=log_level)

    return logger

# from mmcv_custom import load_checkpoint
from mmengine.runner import load_checkpoint
from timm.models.layers import (
    DropPath,
    lecun_normal_,
    to_2tuple,
    trunc_normal_,
)
from timm.models.registry import register_model
from timm.models.vision_transformer import (
    VisionTransformer,
    _cfg,
    _load_weights,
)
from torch import Tensor

from .mamba_simple import Mamba

try:
    from mamba_ssm.ops.triton.layer_norm import (
        RMSNorm,
        layer_norm_fn,
        rms_norm_fn,
    )
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

# from ..builder import BACKBONES
from opencd.registry import MODELS
from typing import (
    Union,
    Tuple,
    Any,
    Callable,
    Iterator,
    Set,
    Optional,
    overload,
    TypeVar,
    Mapping,
    Dict,
    List,
)


class PatchEmbed(nn.Module):
    """2D Image to Patch Embedding"""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        stride=16,
        in_chans=3,
        embed_dim=768,
        norm_layer=None,
        flatten=True,
    ):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (
            (img_size[0] - patch_size[0]) // stride + 1,
            (img_size[1] - patch_size[1]) // stride + 1,
        )
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=stride
        )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], (
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        )
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x


class SwiGLU(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.SiLU,
        drop=0.0,
        norm_layer=nn.LayerNorm,
        subln=False,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.w1 = nn.Linear(in_features, hidden_features)
        self.w2 = nn.Linear(in_features, hidden_features)

        self.act = act_layer()
        self.ffn_ln = norm_layer(hidden_features) if subln else nn.Identity()
        self.w3 = nn.Linear(hidden_features, out_features)

        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x1 = self.w1(x)
        x2 = self.w2(x)
        hidden = self.act(x1) * x2
        x = self.ffn_ln(hidden)
        x = self.w3(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        dim,
        mixer_cls,
        norm_cls=nn.LayerNorm,
        fused_add_norm=False,
        residual_in_fp32=False,
        drop_path=0.0,
    ):
        """
        Simple block wrapping a mixer class with LayerNorm/RMSNorm and residual connection"

        This Block has a slightly different structure compared to a regular
        prenorm Transformer block.
        The standard block is: LN -> MHA/MLP -> Add.
        [Ref: https://arxiv.org/abs/2002.04745]
        Here we have: Add -> LN -> Mixer, returning both
        the hidden_states (output of the mixer) and the residual.
        This is purely for performance reasons, as we can fuse add and LayerNorm.
        The residual needs to be provided (except for the very first block).
        """
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.mixer = mixer_cls(dim)
        self.mlp = SwiGLU(dim, dim * 4 * 2 // 3, subln=False)
        self.drop_path = (
            DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(
        self,
        hidden_states: Tensor,
        residual: Optional[Tensor] = None,
        inference_params=None,
    ):
        hidden_states = hidden_states + self.drop_path(
            self.mixer(
                self.norm1(hidden_states), inference_params=inference_params
            )
        )
        hidden_states = hidden_states + self.drop_path(
            self.mlp(self.norm2(hidden_states))
        )

        return hidden_states

    def allocate_inference_cache(
        self, batch_size, max_seqlen, dtype=None, **kwargs
    ):
        return self.mixer.allocate_inference_cache(
            batch_size, max_seqlen, dtype=dtype, **kwargs
        )


def create_block(
    d_model,
    ssm_cfg=None,
    norm_epsilon=1e-5,
    drop_path=0.0,
    rms_norm=False,
    residual_in_fp32=False,
    fused_add_norm=False,
    layer_idx=None,
    device=None,
    dtype=None,
    if_bimamba=False,
    bimamba_type="none",
    if_devide_out=False,
    init_layer_scale=None,
):
    if if_bimamba:
        bimamba_type = "v1"
    if ssm_cfg is None:
        ssm_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}
    mixer_cls = partial(
        Mamba,
        expand=1,
        layer_idx=layer_idx,
        bimamba_type=bimamba_type,
        if_devide_out=if_devide_out,
        init_layer_scale=init_layer_scale,
        **ssm_cfg,
        **factory_kwargs,
    )
    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm,
        eps=norm_epsilon,
        **factory_kwargs,
    )
    block = Block(
        d_model,
        mixer_cls,
        norm_cls=norm_cls,
        drop_path=drop_path,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
    )
    block.layer_idx = layer_idx
    return block


# https://github.com/huggingface/transformers/blob/c28d04e9e252a1a099944e325685f14d242ecdcd/src/transformers/models/gpt2/modeling_gpt2.py#L454
def _init_weights(
    module,
    n_layer,
    initializer_range=0.02,  # Now only used for embedding layer.
    rescale_prenorm_residual=True,
    n_residuals_per_layer=1,  # Change to 2 if we have MLP
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


def segm_init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        # NOTE conv was left to pytorch default in my original init
        lecun_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)


# @BACKBONES.register_module()
@MODELS.register_module()
class ARM(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        stride=16,
        depth=24,
        embed_dim=192,
        channels=3,
        num_classes=1000,
        ssm_cfg=None,
        drop_rate=0.0,
        drop_path_rate=0.1,
        norm_epsilon: float = 1e-5,
        rms_norm: bool = False,
        initializer_cfg=None,
        fused_add_norm=False,
        residual_in_fp32=False,
        device=None,
        dtype=None,
        ft_seq_len=None,
        pt_hw_seq_len=14,
        if_bidirectional=False,
        final_pool_type="none",
        if_abs_pos_embed=False,
        if_rope=False,
        if_rope_residual=False,
        flip_img_sequences_ratio=-1.0,
        if_bimamba=False,
        bimamba_type="none",
        if_cls_token=False,
        if_devide_out=False,
        init_layer_scale=None,
        use_double_cls_token=False,
        use_middle_cls_token=False,
        global_pool=False,
        **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        # add factory_kwargs into kwargs
        kwargs.update(factory_kwargs)
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.if_bidirectional = if_bidirectional
        self.final_pool_type = final_pool_type
        self.if_abs_pos_embed = if_abs_pos_embed
        self.if_rope = if_rope
        self.if_rope_residual = if_rope_residual
        self.flip_img_sequences_ratio = flip_img_sequences_ratio
        self.if_cls_token = if_cls_token
        self.use_double_cls_token = use_double_cls_token
        self.use_middle_cls_token = use_middle_cls_token
        self.num_tokens = 1 if if_cls_token else 0
        self.global_pool = global_pool
        # pretrain parameters
        self.num_classes = num_classes
        self.d_model = self.num_features = self.embed_dim = (
            embed_dim  # num_features for consistency with other models
        )

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            stride=stride,
            in_chans=channels,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches

        if if_cls_token:
            if use_double_cls_token:
                self.cls_token_head = nn.Parameter(
                    torch.zeros(1, 1, self.embed_dim)
                )
                self.cls_token_tail = nn.Parameter(
                    torch.zeros(1, 1, self.embed_dim)
                )
                self.num_tokens = 2
            else:
                self.cls_token = nn.Parameter(
                    torch.zeros(1, 1, self.embed_dim)
                )
                # self.num_tokens = 1

        if if_abs_pos_embed:
            self.pos_embed = nn.Parameter(
                torch.zeros(1, num_patches + self.num_tokens, self.embed_dim)
            )
            self.pos_drop = nn.Dropout(p=drop_rate)

        if if_rope:
            half_head_dim = embed_dim // 2
            hw_seq_len = img_size // patch_size
            self.rope = VisionRotaryEmbeddingFast(
                dim=half_head_dim,
                pt_seq_len=pt_hw_seq_len,
                ft_seq_len=hw_seq_len,
            )
        self.head = (
            nn.Linear(self.num_features, num_classes)
            if num_classes > 0
            else nn.Identity()
        )

        # TODO: release this comment
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule
        # import ipdb;ipdb.set_trace()
        inter_dpr = [0.0] + dpr
        self.drop_path = (
            DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        )
        # transformer blocks
        self.layers = nn.ModuleList(
            [
                create_block(
                    embed_dim,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    if_bimamba=if_bimamba,
                    bimamba_type=bimamba_type,
                    drop_path=inter_dpr[i],
                    if_devide_out=if_devide_out,
                    init_layer_scale=init_layer_scale,
                    **factory_kwargs,
                )
                for i in range(depth)
            ]
        )

        # fpn-net
        self.fpn1 = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
            nn.GELU(),
            nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
        )

        self.fpn2 = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
        )

        self.fpn3 = nn.Identity()

        self.fpn4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # output head
        self.norm_f = nn.LayerNorm(embed_dim)

        # self.pre_logits = nn.Identity()

        # original init
        self.patch_embed.apply(segm_init_weights)
        self.head.apply(segm_init_weights)
        if if_abs_pos_embed:
            trunc_normal_(self.pos_embed, std=0.02)
        if if_cls_token:
            if use_double_cls_token:
                trunc_normal_(self.cls_token_head, std=0.02)
                trunc_normal_(self.cls_token_tail, std=0.02)
            else:
                trunc_normal_(self.cls_token, std=0.02)

        # mamba init
        self.apply(
            partial(
                _init_weights,
                n_layer=depth,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )

    def init_weights(
        self,
        depth=24,
        pretrained=None,
        if_abs_pos_embed=False,
        if_cls_token=False,
        initializer_cfg=None,
    ):
        """Initialize the weights in backbone.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """

        pretrained = pretrained

        def _init():
            self.patch_embed.apply(segm_init_weights)
            self.head.apply(segm_init_weights)
            if if_abs_pos_embed:
                trunc_normal_(self.pos_embed, std=0.02)
            if if_cls_token:
                if use_double_cls_token:
                    trunc_normal_(self.cls_token_head, std=0.02)
                    trunc_normal_(self.cls_token_tail, std=0.02)
                else:
                    trunc_normal_(self.cls_token, std=0.02)
            # print(self.cls_token)
            # mamba init
            self.apply(
                partial(
                    _init_weights,
                    n_layer=depth,
                    **(initializer_cfg if initializer_cfg is not None else {}),
                )
            )

        if isinstance(pretrained, str):
            _init()
            logger = get_root_logger()
            print(f"load from {pretrained}")
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            _init()
        else:
            raise TypeError("pretrained must be a str or None")

    def allocate_inference_cache(
        self, batch_size, max_seqlen, dtype=None, **kwargs
    ):
        return {
            i: layer.allocate_inference_cache(
                batch_size, max_seqlen, dtype=dtype, **kwargs
            )
            for i, layer in enumerate(self.layers)
        }

    @torch.jit.ignore
    def no_weight_decay(self):
        return {
            "pos_embed",
            "cls_token",
            "dist_token",
            "cls_token_head",
            "cls_token_tail",
        }

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path, prefix=""):
        _load_weights(self, checkpoint_path, prefix)

    def forward_features(
        self,
        x,
        inference_params=None,
        if_random_cls_token_position=False,
        if_random_token_rank=False,
    ):
        # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
        # with slight modifications to add the dist_token
        x = self.patch_embed(x)
        B, M, _ = x.shape
        Hp = Wp = int(M**0.5)

        # 类别标记
        cls_token = self.cls_token.expand(B, -1, -1)
        token_position = M // 2
        x = torch.cat(
            (x[:, :token_position, :], cls_token, x[:, token_position:, :]),
            dim=1,
        )
        M += 1

        # 位置编码
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # mamba impl
        residual = None
        hidden_states = x
        for layer in self.layers:
            # rope about
            if self.if_rope:
                hidden_states = self.rope(hidden_states)
                if residual is not None and self.if_rope_residual:
                    residual = self.rope(residual)
            hidden_states = layer(
                hidden_states, inference_params=inference_params
            )
        hidden_states = self.norm_f(hidden_states)

        mask = torch.ones(
            hidden_states.size(1),
            dtype=torch.bool,
            device=hidden_states.device,
        )
        mask[token_position] = False
        hidden_states = hidden_states[:, mask, :]

        # 转换为2D特征图 [B, C, H, W]
        hidden_states_2d = hidden_states.permute(0, 2, 1).reshape(
            B, -1, Hp, Wp
        )

        # FPN多尺度特征生成
        fpn_output = [
            self.fpn1(hidden_states_2d),  # 4倍上采样
            self.fpn2(hidden_states_2d),  # 2倍上采样
            self.fpn3(hidden_states_2d),  # 原始尺寸
            self.fpn4(hidden_states_2d),  # 2倍下采样
        ]

        return fpn_output

        # if self.global_pool:
        #     return (hidden_states[:, :token_position, :].mean(1)+hidden_states[:, token_position+1:, :].mean(1))/2.
        # else:
        #     return hidden_states[:, token_position, :]

    def forward(
        self,
        x,
        return_features=False,
        inference_params=None,
        if_random_cls_token_position=False,
        if_random_token_rank=False,
    ):
        x = self.forward_features(
            x,
            inference_params,
            if_random_cls_token_position=if_random_cls_token_position,
            if_random_token_rank=if_random_token_rank,
        )

        if return_features:
            return x
        # x = self.head(x)
        return tuple(x)

    def load_state_dict(
        self,
        state_dict: Mapping[str, Any],
        strict: bool = True,
        assign: bool = False,
    ):
        print("Using load_state_dict")
        return super().load_state_dict(state_dict, strict, assign)


def interpolate_pos_embed(model, checkpoint_model):
    if "pos_embed" in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model["pos_embed"]
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int(pos_embed_checkpoint.shape[-2] ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches**0.5)
        # class_token and dist_token are kept unchanged
        extra_tokens = torch.zeros((1, 1, embedding_size))
        print(num_patches, num_extra_tokens, orig_size, new_size)
        if orig_size != new_size:
            print(
                "Position interpolate from %dx%d to %dx%d"
                % (orig_size, orig_size, new_size, new_size)
            )
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint
            pos_tokens = pos_tokens.reshape(
                -1, orig_size, orig_size, embedding_size
            ).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens,
                size=(new_size, new_size),
                mode="bicubic",
                align_corners=False,
            )
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
        else:
            pos_tokens = pos_embed_checkpoint
        B, N, C = pos_tokens.shape
        new_pos_embed = torch.cat(
            (
                pos_tokens[:, : N // 2, :],
                extra_tokens,
                pos_tokens[:, N // 2 :, :],
            ),
            dim=1,
        )
        print("interpolate", new_pos_embed.shape)
        checkpoint_model["pos_embed"] = new_pos_embed
        return checkpoint_model


def re_orgnize(model, ckpt):
    checkpoint_model = ckpt["model"]
    new_dict = {}
    for k, v in checkpoint_model.items():
        if "conv1d" in k:
            new_dict[k.replace("conv1d", "conv1d_b")] = v
            new_dict[k.replace("conv1d", "conv1d_c")] = v
            new_dict[k.replace("conv1d", "conv1d_c_b")] = v
        if "dt_proj" in k:
            new_dict[k.replace("dt_proj", "dt_proj_b")] = v
            new_dict[k.replace("dt_proj", "dt_proj_c")] = v
            new_dict[k.replace("dt_proj", "dt_proj_c_b")] = v
        if "x_proj" in k:
            new_dict[k.replace("x_proj", "x_proj_b")] = v
            new_dict[k.replace("x_proj", "x_proj_c")] = v
            new_dict[k.replace("x_proj", "x_proj_c_b")] = v
        if "A" in k:
            new_dict[k.replace("A", "A_b")] = v
            new_dict[k.replace("A", "A_c")] = v
            new_dict[k.replace("A", "A_c_b")] = v
        if "D" in k:
            new_dict[k.replace("D", "D_b")] = v
            new_dict[k.replace("D", "D_c")] = v
            new_dict[k.replace("D", "D_c_b")] = v
        new_dict[k] = v
    # interpolate position embedding
    # !
    new_dict = interpolate_pos_embed(model, new_dict)

    # load pre-trained model
    print(new_dict["pos_embed"].shape, model.state_dict()["pos_embed"].shape)
    msg = model.load_state_dict(new_dict, strict=False)
    print("#####msg", msg)

@register_model
@MODELS.register_module()
def arm_tiny_pz16(pretrained=False, **kwargs):
    model = ARM(
        patch_size=16,
        embed_dim=384,
        depth=12,
        rms_norm=True,
        residual_in_fp32=True,
        fused_add_norm=True,
        final_pool_type="mean",
        if_abs_pos_embed=True,
        if_rope=False,
        if_rope_residual=False,
        bimamba_type="v3",
        if_cls_token=True,
        if_devide_out=True,
        use_middle_cls_token=True,
        **kwargs,
    )
    model.default_cfg = _cfg()
    print("##pretrained bool:", pretrained)
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url=pretrained, map_location="cpu", check_hash=False
        )
        # print("##########1", model.state_dict().keys())
        # print("##########1", checkpoint.keys())
        re_orgnize(model, checkpoint)
        # print("##########2", model)
        # print("##########2", checkpoint.keys())
        # model.load_state_dict(checkpoint["model"])
    return model


# @BACKBONES.register_module()
@register_model
@MODELS.register_module()
def arm_base_pz16(pretrained=False, **kwargs):
    model = ARM(
        patch_size=16,
        embed_dim=768,
        depth=12,
        rms_norm=True,
        residual_in_fp32=True,
        fused_add_norm=True,
        final_pool_type="mean",
        if_abs_pos_embed=True,
        if_rope=False,
        if_rope_residual=False,
        bimamba_type="v3",
        if_cls_token=True,
        if_devide_out=True,
        use_middle_cls_token=True,
        **kwargs,
    )
    model.default_cfg = _cfg()
    print("##pretrained bool:", pretrained)
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url=pretrained, map_location="cpu", check_hash=False
        )
        # print("##########1", model.state_dict().keys())
        # print("##########1", checkpoint.keys())
        re_orgnize(model, checkpoint)
        # print("##########2", model)
        # print("##########2", checkpoint.keys())
        # model.load_state_dict(checkpoint["model"])
    return model


# @BACKBONES.register_module()
@register_model
@MODELS.register_module()
def arm_large_pz16(pretrained=False, **kwargs):
    model = ARM(
        patch_size=16,
        embed_dim=1024,
        depth=24,
        rms_norm=True,
        residual_in_fp32=True,
        fused_add_norm=True,
        final_pool_type="mean",
        if_abs_pos_embed=True,
        if_rope=False,
        if_rope_residual=False,
        bimamba_type="v3",
        if_cls_token=True,
        if_devide_out=True,
        use_middle_cls_token=True,
        **kwargs,
    )
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url=pretrained, map_location="cpu", check_hash=False
        )
        # print("##########1", model.state_dict().keys())
        # print("##########1", checkpoint.keys())
        re_orgnize(model, checkpoint)
        # print("##########2", model)
        # print("##########2", checkpoint.keys())
        # model.load_state_dict(checkpoint["model"])
    return model


# @BACKBONES.register_module()
@register_model
@MODELS.register_module()
def arm_huge_pz16(pretrained=False, **kwargs):
    model = ARM(
        patch_size=16,
        embed_dim=1536,
        depth=24,
        rms_norm=True,
        residual_in_fp32=True,
        fused_add_norm=True,
        final_pool_type="mean",
        if_abs_pos_embed=True,
        if_rope=False,
        if_rope_residual=False,
        bimamba_type="v3",
        if_cls_token=True,
        if_devide_out=True,
        use_middle_cls_token=True,
        **kwargs,
    )
    model.default_cfg = _cfg()
    if pretrained:
        print("trying to load ckpt")
        checkpoint = torch.hub.load_state_dict_from_url(
            url=pretrained, map_location="cpu", check_hash=False
        )
        # print("##########1", model.state_dict().keys())
        # print("##########1", checkpoint.keys())
        re_orgnize(model, checkpoint)
        # print("##########2", model)
        # print("##########2", checkpoint.keys())
        # model.load_state_dict(checkpoint["model"])
    return model

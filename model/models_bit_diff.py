import torch
import torch.nn as nn
import copy
import math
from einops import rearrange
from utils import * # 确保 utils 在同一目录下

# 导入标准 Mamba
from mamba_ssm.modules.mamba_simple import Mamba as ViM

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class BitDiffPredictorTCN(nn.Module):
    def __init__(self, args, causal=False):
        super(BitDiffPredictorTCN, self).__init__()

        self.ms_tcn = DiffMultiStageModel(
            args.layer_type,
            args.kernel_size,
            args.num_stages,
            args.num_layers,
            args.model_dim,
            args.input_dim + 2 * args.num_classes, 
            args.num_classes,
            args.channel_dropout_prob,
            args.use_features,
        )

        self.use_inp_ch_dropout = args.use_inp_ch_dropout
        if args.use_inp_ch_dropout:
            self.channel_dropout = torch.nn.Dropout1d(args.channel_dropout_prob)

    def forward(self, x, t, stage_masks, obs_cond=None, self_cond=None):
        # arange
        x = rearrange(x, "b t c -> b c t")
        obs_cond = rearrange(obs_cond, "b t c -> b c t")
        self_cond = rearrange(self_cond, "b t c -> b c t")
        stage_masks = [rearrange(mask, "b t c -> b c t") for mask in stage_masks]

        if self.use_inp_ch_dropout:
            x = self.channel_dropout(x)

        # condition on input
        x = torch.cat((x, obs_cond), dim=1)
        x = torch.cat((x, self_cond), dim=1)

        frame_wise_pred, _ = self.ms_tcn(x, t, stage_masks)
        frame_wise_pred = rearrange(frame_wise_pred, "s b c t -> s b t c")
        return frame_wise_pred

class DiffMultiStageModel(nn.Module):
    def __init__(
        self,
        layer_type,
        kernel_size,
        num_stages,
        num_layers,
        num_f_maps,
        dim,
        num_classes,
        dropout,
        use_features=False,
    ):
        super(DiffMultiStageModel, self).__init__()
        self.stage1 = DiffSingleStageModel(
            layer_type,
            kernel_size,
            num_layers,
            num_f_maps,
            dim,
            num_classes,
            dropout,
        )

    def forward(self, x, t, stage_masks):
        out, out_features = self.stage1(x, t, stage_masks[0])
        outputs = out.unsqueeze(0)
        return outputs, out_features

class DiffSingleStageModel(nn.Module):
    def __init__(
        self,
        layer_type,
        kernel_size,
        num_layers,
        num_f_maps,
        dim,
        num_classes,
        dropout,
    ):
        super(DiffSingleStageModel, self).__init__()

        self.layer_types = {
            "mamba": DiffMambaResidualLayer,
        }

        self.conv_1x1 = nn.Conv1d(dim, num_f_maps, 1)

        # time cond
        time_dim = num_f_maps * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(num_f_maps),
            nn.Linear(num_f_maps, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        # MAMBA
        if layer_type in ["mamba"]:
            self.layers = []
            for i in range(num_layers):
                self.layers.append(
                    copy.deepcopy(
                        self.layer_types[layer_type](
                            kernel_size,
                            num_f_maps,
                            time_dim,
                            dropout,
                            'sum',
                            bimamba=True, # 启用我们手动实现的双向
                        )
                    )
                )

        print(f"Total layers: {len(self.layers)}")
        self.layers = nn.ModuleList(self.layers)
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x, t, mask):
        out = self.conv_1x1(x) * mask
        time = self.time_mlp(t)

        for layer in self.layers:
            out = layer(out, time, mask)

        out_features = out * mask
        out_logits = self.conv_out(out) * mask
        return out_logits, out_features

# ==========================================
# 修复后的 Mamba Layer (适配标准库 + 手动双向)
# ==========================================
class DiffMambaResidualLayer(nn.Module):
    def __init__(
        self,
        kernel_size,
        out_channels,
        time_channels=-1,
        dropout=0.2,
        accum='sum',
        bimamba=True,
    ):
        super(DiffMambaResidualLayer, self).__init__()
        
        self.bimamba = bimamba
        self.accum = accum

        # 1. 正向 Mamba
        # 标准库参数: d_model, d_conv, use_fast_path
        # 注意: 去掉了 bimamba, dropout, accum 参数
        self.mamba = ViM(
            d_model=out_channels,
            d_conv=kernel_size,
            use_fast_path=True,
        )
        
        # 2. 反向 Mamba (如果启用)
        if self.bimamba:
            self.mamba_inv = ViM(
                d_model=out_channels,
                d_conv=kernel_size,
                use_fast_path=True,
            )

        # DropPath 需要 utils.py 中有 AffineDropPath 实现
        # 如果没有，可以使用简单的 nn.Dropout 替代，或者确保 utils 存在
        try:
            self.drop_path = AffineDropPath(out_channels, drop_prob=dropout)
        except NameError:
             # Fallback if utils not imported or AffineDropPath missing
             self.drop_path = nn.Dropout(dropout)
             
        self.norm = nn.LayerNorm(out_channels)

        # out block
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout(dropout)

        # Time Net
        self.time_channels = time_channels
        if time_channels > 0:
            self.time_mlp = nn.Sequential(
                nn.SiLU(), nn.Linear(time_channels, out_channels * 2)
            )

    def forward(self, x, t, mask):
        # x: [B, C, T] (Conv1d format)
        # mask: [B, C, T]
        
        # Norm & Permute for Mamba [B, T, C]
        x_in = x.permute(0, 2, 1) # B T C
        mask_in = mask.permute(0, 2, 1) # B T C
        
        mamba_in = self.norm(x_in) * mask_in

        # --- Forward Pass ---
        fwd_out = self.mamba(mamba_in) # B T C

        # --- Backward Pass (Manual Bidirectional) ---
        if self.bimamba:
            # 翻转时间维度
            inv_in = torch.flip(mamba_in, dims=[1]) 
            inv_out = self.mamba_inv(inv_in)
            # 翻转回来
            inv_out = torch.flip(inv_out, dims=[1])
            
            if self.accum == 'sum':
                mamba_out = fwd_out + inv_out
            elif self.accum == 'mean':
                mamba_out = (fwd_out + inv_out) / 2
            else:
                mamba_out = fwd_out + inv_out
        else:
            mamba_out = fwd_out

        # Permute back to [B, C, T]
        mamba_out = mamba_out.permute(0, 2, 1)
        
        # DropPath & Mask
        # 注意: 这里的 mamba_out 是 [B, C, T], drop_path 实现可能需要适配
        # 假设 AffineDropPath 接受 (B, C, T) 或者 standard dropout
        mamba_out = self.drop_path(mamba_out) * mask

        # Conv projection
        mamba_out = self.conv_1x1(mamba_out) * mask
        mamba_out = self.dropout(mamba_out)

        # Time Conditioning
        if self.time_channels > 0:
            time_scale, time_shift = self.time_mlp(t).chunk(2, dim=1)
            time_scale = rearrange(time_scale, "b d -> b d 1")
            time_shift = rearrange(time_shift, "b d -> b d 1")
            mamba_out = mamba_out * (time_scale + 1) + time_shift

        return (x + mamba_out) * mask
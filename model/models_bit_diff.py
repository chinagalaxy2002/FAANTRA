import torch
import torch.nn as nn
import copy
import math
from einops import rearrange

# ==========================================
# [修复] 正确的 try-except 格式
# ==========================================
try:
    from utils import *
except ImportError:
    pass

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

        # 这里的 num_f_maps 使用 args.model_dim (即 hidden_dim)
        self.ms_tcn = DiffMultiStageModel(
            args.layer_type,
            args.kernel_size,
            args.num_stages,
            args.num_layers,
            args.model_dim,      # num_f_maps (内部特征维度)
            args.num_classes,    # x_dim (未来帧输入维度)
            args.input_dim,      # obs_dim (过去帧输入维度)
            args.num_classes,    # output_classes
            args.channel_dropout_prob,
            args.use_features,
        )

        self.use_inp_ch_dropout = args.use_inp_ch_dropout
        if args.use_inp_ch_dropout:
            self.channel_dropout = torch.nn.Dropout1d(args.channel_dropout_prob)

    def forward(self, x, t, stage_masks, obs_cond=None, self_cond=None):
        # [修改] 输入处理
        # x: [B, T, C_out] -> [B, C_out, T]
        x = rearrange(x, "b t c -> b c t")
        
        # [关键修复] Mask 处理
        # stage_masks 中的 mask 也是 [B, T, 1]，需要转为 [B, 1, T] 以匹配 concatenation
        if stage_masks is not None:
            stage_masks = [rearrange(m, "b t c -> b c t") for m in stage_masks]
        
        # obs_cond: [B, S, C_in] -> [B, C_in, S]
        if obs_cond is not None:
            obs_cond = rearrange(obs_cond, "b t c -> b c t")
        
        # self_cond: 暂不处理复杂拼接，保持原样或忽略
        if self_cond is not None:
             self_cond = rearrange(self_cond, "b t c -> b c t")

        if self.use_inp_ch_dropout:
            x = self.channel_dropout(x)

        # [修改] 不再在这里拼接 channel，而是传入 ms_tcn 内部处理
        frame_wise_pred, _ = self.ms_tcn(x, t, stage_masks, obs_cond=obs_cond)
        
        # [修改] 输出重排回 [B, T, C]
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
        x_dim,       
        obs_dim,     
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
            x_dim,
            obs_dim,
            num_classes,
            dropout,
        )

    def forward(self, x, t, stage_masks, obs_cond=None):
        # [修改] 传递 obs_cond
        out, out_features = self.stage1(x, t, stage_masks[0], obs_cond=obs_cond)
        outputs = out.unsqueeze(0)
        return outputs, out_features

class DiffSingleStageModel(nn.Module):
    def __init__(
        self,
        layer_type,
        kernel_size,
        num_layers,
        num_f_maps,
        x_dim,       
        obs_dim,     
        num_classes,
        dropout,
    ):
        super(DiffSingleStageModel, self).__init__()

        self.layer_types = {
            "mamba": DiffMambaResidualLayer,
        }

        # [修改] 分别定义投影层
        self.x_proj = nn.Conv1d(x_dim, num_f_maps, 1)
        self.obs_proj = nn.Conv1d(obs_dim, num_f_maps, 1)

        # time cond
        time_dim = num_f_maps * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(num_f_maps),
            nn.Linear(num_f_maps, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        # MAMBA Layers
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
                            bimamba=True, 
                        )
                    )
                )

        print(f"Total layers: {len(self.layers)}")
        self.layers = nn.ModuleList(self.layers)
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x, t, mask, obs_cond=None):
        # x: [B, C_x, T_future] (Noisy input)
        # obs_cond: [B, C_obs, T_past] (Condition)
        # mask: [B, 1, T_future]
        
        # 1. 投影
        x_emb = self.x_proj(x) # [B, D, T_future]
        
        # 2. Time-Concat 拼接
        if obs_cond is not None:
            obs_emb = self.obs_proj(obs_cond) # [B, D, T_past]
            
            # 在时间维度拼接 (dim=2) -> [B, D, T_past + T_future]
            h = torch.cat([obs_emb, x_emb], dim=2)
            
            # 处理 Mask
            # 构造 obs 的 mask (假设全 1)
            B, _, T_past = obs_emb.shape
            mask_obs = torch.ones((B, 1, T_past), device=x.device)
            
            # 拼接 Mask -> [B, 1, T_past + T_future]
            mask_combined = torch.cat([mask_obs, mask], dim=2)
        else:
            h = x_emb
            mask_combined = mask

        # 3. Time Embedding
        time = self.time_mlp(t)

        # 4. Pass through Layers
        out = h
        for layer in self.layers:
            out = layer(out, time, mask_combined)

        # 5. 切片 (Slice)
        # 只保留对应 Future 的部分用于预测和 Loss 计算
        T_future = x.shape[2]
        out_future = out[:, :, -T_future:] # 取最后 T_future 帧

        # 6. Output Projection
        out_features = out_future * mask
        out_logits = self.conv_out(out_future) * mask
        
        return out_logits, out_features

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

        self.mamba = ViM(
            d_model=out_channels,
            d_conv=kernel_size,
            use_fast_path=True,
        )
        
        if self.bimamba:
            self.mamba_inv = ViM(
                d_model=out_channels,
                d_conv=kernel_size,
                use_fast_path=True,
            )

        try:
            self.drop_path = AffineDropPath(out_channels, drop_prob=dropout)
        except NameError:
             self.drop_path = nn.Dropout(dropout)
             
        self.norm = nn.LayerNorm(out_channels)
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout(dropout)

        self.time_channels = time_channels
        if time_channels > 0:
            self.time_mlp = nn.Sequential(
                nn.SiLU(), nn.Linear(time_channels, out_channels * 2)
            )

    def forward(self, x, t, mask):
        # x: [B, C, T]
        x_in = x.permute(0, 2, 1) # B T C
        mask_in = mask.permute(0, 2, 1) # B T C
        
        mamba_in = self.norm(x_in) * mask_in

        fwd_out = self.mamba(mamba_in) 

        if self.bimamba:
            inv_in = torch.flip(mamba_in, dims=[1]) 
            inv_out = self.mamba_inv(inv_in)
            inv_out = torch.flip(inv_out, dims=[1])
            
            if self.accum == 'sum':
                mamba_out = fwd_out + inv_out
            elif self.accum == 'mean':
                mamba_out = (fwd_out + inv_out) / 2
            else:
                mamba_out = fwd_out + inv_out
        else:
            mamba_out = fwd_out

        mamba_out = mamba_out.permute(0, 2, 1)
        mamba_out = self.drop_path(mamba_out) * mask
        mamba_out = self.conv_1x1(mamba_out) * mask
        mamba_out = self.dropout(mamba_out)

        if self.time_channels > 0:
            time_scale, time_shift = self.time_mlp(t).chunk(2, dim=1)
            time_scale = rearrange(time_scale, "b d -> b d 1")
            time_shift = rearrange(time_shift, "b d -> b d 1")
            mamba_out = mamba_out * (time_scale + 1) + time_shift

        return (x + mamba_out) * mask
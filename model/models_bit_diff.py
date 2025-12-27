import torch
import torch.nn as nn
import copy
import math
from einops import rearrange
from utils import * # 导入标准 Mamba
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

        # ============================================================
        # [核心修改 1] 定义投影层
        # 目的：将不同维度的 Future(动作) 和 Past(特征) 映射到统一维度，以便在时间轴上拼接
        # ============================================================
        # 输入: Future (Batch, 15, T_fut) -> 投影到 model_dim
        self.input_proj = nn.Conv1d(args.num_classes, args.model_dim, 1)
        # 输入: Past (Batch, 256, T_past) -> 投影到 model_dim
        self.cond_proj = nn.Conv1d(args.input_dim, args.model_dim, 1)

        self.ms_tcn = DiffMultiStageModel(
            args.layer_type,
            args.kernel_size,
            args.num_stages,
            args.num_layers,
            args.model_dim, # 隐藏层维度
            args.model_dim, # [核心修改 2] 输入维度现在统一为 model_dim
            args.num_classes,
            args.channel_dropout_prob,
            args.use_features,
        )

        self.use_inp_ch_dropout = args.use_inp_ch_dropout
        if args.use_inp_ch_dropout:
            self.channel_dropout = torch.nn.Dropout1d(args.channel_dropout_prob)

    def forward(self, x, t, stage_masks, obs_cond=None, self_cond=None):
        # x: [B, T_fut, 15] (Noisy Future)
        # obs_cond: [B, T_past, 256] (Clean Past Features)
        
        # 1. 调整维度 [B, T, C] -> [B, C, T]
        x = rearrange(x, "b t c -> b c t")
        if obs_cond is not None:
            # obs_cond 此时是原始长度，不再是被强行 interpolate 过的
            obs_cond = rearrange(obs_cond, "b t c -> b c t")
        
        # 2. 独立投影
        x_emb = self.input_proj(x) # [B, model_dim, T_fut]
        
        if obs_cond is not None:
            cond_emb = self.cond_proj(obs_cond) # [B, model_dim, T_past]
            T_past = cond_emb.shape[-1]
            
            # ============================================================
            # [核心修改 3] 时间维度拼接 (Temporal Concatenation)
            # 序列变成了: [Past, Future]
            # Mamba 将利用 Past 的隐状态来去噪 Future
            # ============================================================
            x_combined = torch.cat((cond_emb, x_emb), dim=-1) # [B, model_dim, T_past + T_fut]
            
            # 3. 处理 Mask
            # stage_masks 对应 x (Future), 我们需要为 Cond (Past) 补充全 1 Mask
            B = x.shape[0]
            device = x.device
            # 过去的帧是真实存在的，Mask 为 1
            cond_mask = torch.ones((B, 1, T_past), device=device)
            
            new_stage_masks = []
            for mask in stage_masks:
                mask = rearrange(mask, "b t c -> b c t")
                # 拼接 Mask: [1...1, mask_future...]
                full_mask = torch.cat((cond_mask, mask), dim=-1) 
                new_stage_masks.append(full_mask)
            
            input_tensor = x_combined
            masks_to_pass = new_stage_masks
        else:
            # Fallback (通常不会走到这里)
            input_tensor = x_emb
            masks_to_pass = [rearrange(mask, "b t c -> b c t") for mask in stage_masks]
            T_past = 0

        if self.use_inp_ch_dropout:
            input_tensor = self.channel_dropout(input_tensor)

        # 4. 送入骨干网络 Mamba
        # out: [Samples, B, model_dim, T_total]
        out, out_features = self.ms_tcn(input_tensor, t, masks_to_pass)
        
        # ============================================================
        # [核心修改 4] 输出切片 (Output Slicing)
        # 我们只关心预测的未来部分，切掉前面的 Past 部分
        # ============================================================
        if T_past > 0:
            out = out[..., T_past:] # 取后 T_fut 帧
        
        # 恢复维度 [S, B, T_fut, 15]
        out = rearrange(out, "s b c t -> s b t c")
        return out

class DiffMultiStageModel(nn.Module):
    def __init__(
        self,
        layer_type,
        kernel_size,
        num_stages,
        num_layers,
        num_f_maps,
        dim, # input channels
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

        # 这里的 dim 现在是 model_dim
        self.conv_1x1 = nn.Conv1d(dim, num_f_maps, 1)

        # time cond
        time_dim = num_f_maps * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(num_f_maps),
            nn.Linear(num_f_maps, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        # MAMBA Layer Stacking
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
        # 输出头：预测 num_classes (15维)
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x, t, mask):
        out = self.conv_1x1(x) * mask
        time = self.time_mlp(t)

        for layer in self.layers:
            out = layer(out, time, mask)

        out_features = out * mask
        out_logits = self.conv_out(out) * mask
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

        # Standard Mamba
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
        x_in = x.permute(0, 2, 1) 
        mask_in = mask.permute(0, 2, 1) 
        
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
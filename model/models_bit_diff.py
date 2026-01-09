import torch
import torch.nn as nn
import copy
import math
from einops import rearrange

# ==========================================
# v4 - Offset Binning Version
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

        # [参数解析]
        self.num_actions = args.num_classes       
        self.offset_dim = getattr(args, 'offset_dim', 0)         
        self.actionness_dim = getattr(args, 'actionness_dim', 0) 
        
        # [新增] Offset Bin 数量，默认64
        self.num_offset_bins = getattr(args, 'num_offset_bins', 64)

        # 这里的 num_f_maps 使用 args.model_dim (即 hidden_dim)
        self.ms_tcn = DiffMultiStageModel(
            args.layer_type,
            args.kernel_size,
            args.num_stages,
            args.num_layers,
            args.model_dim,      # num_f_maps (内部特征维度)
            self.num_actions,    # x_dim (未来帧输入维度，即纯类别维度)
            args.input_dim,      # obs_dim (过去帧输入维度)
            self.num_actions,    # num_classes (Diff Head 输出维度)
            args.channel_dropout_prob,
            args.use_features,
            offset_dim=self.offset_dim,        
            actionness_dim=self.actionness_dim,
            num_offset_bins=self.num_offset_bins # [新增] 传递 Bin 数量
        )

        self.use_inp_ch_dropout = args.use_inp_ch_dropout
        if args.use_inp_ch_dropout:
            self.channel_dropout = torch.nn.Dropout1d(args.channel_dropout_prob)

    def forward(self, x, t, stage_masks, obs_cond=None, self_cond=None):
        # [修改] 输入处理
        # x: [B, T, C_out] -> [B, C_out, T]
        x = rearrange(x, "b t c -> b c t")
        
        # [关键修复] Mask 处理
        if stage_masks is not None:
            new_masks = []
            for m in stage_masks:
                # m: [B, T, C]
                if m.shape[-1] > 1:
                    m = m[..., 0:1] # 只取第一个 channel -> [B, T, 1]
                
                m = rearrange(m, "b t c -> b c t") # -> [B, 1, T]
                new_masks.append(m)
            stage_masks = new_masks
        
        # obs_cond: [B, S, C_in] -> [B, C_in, S]
        if obs_cond is not None:
            obs_cond = rearrange(obs_cond, "b t c -> b c t")
        
        # self_cond
        if self_cond is not None:
             self_cond = rearrange(self_cond, "b t c -> b c t")

        if self.use_inp_ch_dropout:
            x = self.channel_dropout(x)

        # 获取输出字典
        outputs_dict = self.ms_tcn(x, t, stage_masks, obs_cond=obs_cond)
        
        # 输出重排回 [S, B, T, C]
        res = {}
        keys_to_process = ['action', 'offset', 'actionness', 'features']
        
        for key in keys_to_process:
            if key in outputs_dict and outputs_dict[key] is not None:
                # [s b c t] -> [s b t c]
                res[key] = rearrange(outputs_dict[key], "s b c t -> s b t c")
            else:
                res[key] = None
                
        return res

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
        offset_dim=0,      
        actionness_dim=0,
        num_offset_bins=64 # [新增]
    ):
        super(DiffMultiStageModel, self).__init__()
        # 目前只支持单 Stage
        self.stage1 = DiffSingleStageModel(
            layer_type,
            kernel_size,
            num_layers,
            num_f_maps,
            x_dim,
            obs_dim,
            num_classes,
            dropout,
            offset_dim=offset_dim,         
            actionness_dim=actionness_dim,
            num_offset_bins=num_offset_bins # [新增]
        )

    def forward(self, x, t, stage_masks, obs_cond=None):
        out_dict = self.stage1(x, t, stage_masks[0], obs_cond=obs_cond)
        
        # 增加 stage 维度
        outputs = {}
        for k, v in out_dict.items():
            if v is not None:
                outputs[k] = v.unsqueeze(0) 
            else:
                outputs[k] = None
                
        return outputs

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
        offset_dim=0,    
        actionness_dim=0,
        num_offset_bins=64 # [新增]
    ):
        super(DiffSingleStageModel, self).__init__()

        self.layer_types = {
            "mamba": DiffMambaResidualLayer,
        }

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

        # [修改] 定义分离的 Head
        
        # 1. Class Head (Diffusion)
        self.cls_head = nn.Conv1d(num_f_maps, num_classes, 1)
        
        # 2. Offset Head (改为分类 Head)
        self.offset_dim = offset_dim
        if offset_dim > 0:
            self.offset_head = nn.Sequential(
                nn.Conv1d(num_f_maps, num_f_maps, 1),
                nn.GELU(),
                # [关键修改] 输出维度变成 num_offset_bins (例如 64)
                nn.Conv1d(num_f_maps, num_offset_bins, 1) 
            )
        else:
            self.offset_head = None
            
        # 3. Actionness Head (保持回归)
        self.actionness_dim = actionness_dim
        if actionness_dim > 0:
            self.actionness_head = nn.Sequential(
                nn.Conv1d(num_f_maps, num_f_maps, 1),
                nn.GELU(),
                nn.Conv1d(num_f_maps, actionness_dim, 1)
            )
        else:
            self.actionness_head = None

    def forward(self, x, t, mask, obs_cond=None):
        # x: [B, C_x, T_future]
        # obs_cond: [B, C_obs, T_past]
        
        # 1. 投影
        x_emb = self.x_proj(x)
        
        # 2. Time-Concat 拼接
        if obs_cond is not None:
            obs_emb = self.obs_proj(obs_cond)
            h = torch.cat([obs_emb, x_emb], dim=2)
            
            # 处理 Mask
            B, _, T_past = obs_emb.shape
            mask_obs = torch.ones((B, 1, T_past), device=x.device)
            mask_combined = torch.cat([mask_obs, mask], dim=2)
        else:
            h = x_emb
            mask_combined = mask

        # 3. Time Embedding
        time = self.time_mlp(t)

        # 4. Layers
        out = h
        for layer in self.layers:
            out = layer(out, time, mask_combined)

        # 5. 切片
        T_future = x.shape[2]
        out_future = out[:, :, -T_future:] 

        # 6. Output Projection
        out_features = out_future * mask
        
        # (1) Class Prediction
        out_cls = self.cls_head(out_future) * mask
        
        # (2) Offset Prediction (logits for bins)
        out_off = None
        if self.offset_head is not None:
            out_off = self.offset_head(out_future) * mask
            
        # (3) Actionness Prediction
        out_act = None
        if self.actionness_head is not None:
            out_act = self.actionness_head(out_future) * mask
        
        return {
            "action": out_cls,
            "offset": out_off,
            "actionness": out_act,
            "features": out_features
        }

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
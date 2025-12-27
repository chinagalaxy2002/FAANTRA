import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import math
import os
import sys
import timm
import torchvision.transforms as T
from einops import repeat, rearrange

# ==========================================
# 导入模块
# ==========================================
try:
    from bit_diffusion import GaussianBitDiffusion
    from models_bit_diff import BitDiffPredictorTCN
    from model.T_Deed_Modules.shift import make_temporal_shift
except ImportError:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    try:
        from bit_diffusion import GaussianBitDiffusion
        from models_bit_diff import BitDiffPredictorTCN
        from model.T_Deed_Modules.shift import make_temporal_shift
    except ImportError as e:
        print("Error: 无法导入 bit_diffusion, models_bit_diff 或 shift 模块。")
        raise e

class DiffusionConfig:
    def __init__(self, args, input_dim, num_classes):
        self.layer_type = "mamba"
        self.kernel_size = 3
        self.num_stages = 1
        self.num_layers = args.num_encoder_layers if hasattr(args, 'num_encoder_layers') else 4
        self.model_dim = args.hidden_dim
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.channel_dropout_prob = 0.1
        self.use_features = True
        self.use_inp_ch_dropout = False

class FUTR(nn.Module):
    def __init__(self, n_class, hidden_dim, src_pad_idx, device, args, n_query=8, n_head=8,
                 num_encoder_layers=6, num_decoder_layers=6, src_attn_mask=None, tgt_attn_mask=None):
        super().__init__()
        
        self.device = device
        self.n_query = n_query
        self.n_class = n_class
        self.hidden_dim = hidden_dim
        self.args = args
        self.src_pad_idx = src_pad_idx
        
        # 1. Backbone
        self.feature_arch = args.feature_arch
        if self.feature_arch.startswith(('rny002', 'rny004', 'rny006', 'rny008')):
            self.features = timm.create_model({
                'rny002': 'regnety_002',
                'rny004': 'regnety_004',
                'rny006': 'regnety_006',
                'rny008': 'regnety_008',
            }[self.feature_arch.rsplit('_', 1)[0]], pretrained=True)
            feat_dim = self.features.head.fc.in_features
            self.features.head.fc = nn.Identity()
            self.input_dim = feat_dim
        else:
            raise NotImplementedError(f"Architecture {args.feature_arch} not supported")
            
        max_obs_len = int(args.clip_len*args.cheating_range[1])-int(args.clip_len*args.cheating_range[0]) if args.cheating_dataset else int(args.clip_len*max(args.obs_perc))
        if self.feature_arch.endswith('_gsm'):
            make_temporal_shift(self.features, max_obs_len, mode='gsm')
        elif self.feature_arch.endswith('_gsf'):
            make_temporal_shift(self.features, max_obs_len, mode='gsf')

        # 2. Projection
        self.input_proj = nn.Linear(self.input_dim, hidden_dim)
        self.relu = nn.ReLU()

        if args.seg:
            self.fc_seg = nn.Linear(hidden_dim, n_class)

        # 3. Diffusion Mamba Setup
        self.diff_out_dim = n_class 
        self.offset_dim = 1
        self.diff_out_dim += self.offset_dim
        
        if args.actionness:
            self.actionness_dim = 1
            self.diff_out_dim += self.actionness_dim
        else:
            self.actionness_dim = 0

        # 注意：这里传给 Config 的 input_dim 是 hidden_dim (RegNet 投影后的维度)
        diff_cfg = DiffusionConfig(args, hidden_dim, self.diff_out_dim)
        self.denoise_model = BitDiffPredictorTCN(diff_cfg)

        sampling_steps = getattr(args, 'ddim_timesteps', 50) 
        
        # [保留修改] Timesteps = 1000
        self.diffusion = GaussianBitDiffusion(
            model=self.denoise_model,
            condition_x0=False,
            num_classes=self.diff_out_dim,
            timesteps=1000,          
            ddim_timesteps=sampling_steps, 
            loss_type="l2",
            objective="pred_x0",
            beta_schedule="cosine"
        )

        # 4. Augmentation
        self.augmentation = T.Compose([
            T.RandomApply([T.ColorJitter(hue=0.2)], p=0.25),
            T.RandomApply([T.ColorJitter(saturation=(0.7, 1.2))], p=0.25),
            T.RandomApply([T.ColorJitter(brightness=(0.7, 1.2))], p=0.25),
            T.RandomApply([T.ColorJitter(contrast=(0.7, 1.2))], p=0.25),
            T.RandomApply([T.GaussianBlur(5)], p=0.25),
            T.RandomHorizontalFlip(),
        ])
        
        self.standarization = T.Compose([
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])

    def augment(self, x):
        for i in range(x.shape[0]):
            x[i] = self.augmentation(x[i])
        return x

    def standarize(self, x):
        for i in range(x.shape[0]):
            x[i] = self.standarization(x[i])
        return x

    def forward(self, inputs, mode='train'):
        targets_dict = None
        
        if mode == 'train':
            if isinstance(inputs, tuple) or isinstance(inputs, list):
                src = inputs[0]
                if len(inputs) > 1 and isinstance(inputs[1], dict):
                    targets_dict = inputs[1]
                elif len(inputs) > 1:
                    src_label = inputs[1] 
            else:
                src = inputs
        else:
            src = inputs

        B, S, C, H, W = src.size()
        src = src / 255.0
        
        if mode == "train":
            src = self.augment(src)
        
        src = self.standarize(src)
        
        # 特征提取
        features = self.features(src.view(-1, C, H, W)).reshape(B, S, self.input_dim)
        obs_feat = self.relu(self.input_proj(features))

        output = dict()
        if self.args.seg:
            output['seg'] = self.fc_seg(obs_feat)

        # ============================================================
        # [核心修改] 移除 Interpolate，直接使用 obs_feat 作为 Condition
        # obs_feat 维度: [Batch, T_past, HiddenDim]
        # ============================================================
        obs_cond = obs_feat 
        
        # Mask 准备 (仅针对 Future 部分)
        mask_past = torch.ones((B, self.n_query, 1), device=self.device)
        masks_stages = [torch.ones((B, self.n_query, 1), device=self.device)]
        
        norm_factor = float(self.args.clip_len)

        if mode == 'train':
            if targets_dict is None or 'action' not in targets_dict:
                raise ValueError("Diffusion Training requires targets_dict.")

            gt_action = targets_dict['action'].long()
            gt_offset = targets_dict['offset']
            
            # Mask Logic
            is_valid = (gt_action != self.src_pad_idx)
            mask_past = is_valid.unsqueeze(-1).float()
            masks_stages = [mask_past] 
            
            gt_action_safe = gt_action.clone()
            gt_action_safe[~is_valid] = 0 
            
            # [保留修改] 数据中心化映射 [-1, 1]
            x_0_cls = F.one_hot(gt_action_safe, num_classes=self.n_class).float()
            x_0_cls = x_0_cls * 2.0 - 1.0 
            
            gt_offset_norm = gt_offset / norm_factor
            x_0_off = (gt_offset_norm.unsqueeze(-1) * 2.0) - 1.0
            
            x_0_parts = [x_0_cls, x_0_off]
            
            if self.args.actionness and 'actionness' in targets_dict:
                gt_act = targets_dict['actionness'].unsqueeze(-1)
                gt_act = gt_act * 2.0 - 1.0 
                x_0_parts.append(gt_act)
            
            x_0 = torch.cat(x_0_parts, dim=-1)

            t = torch.randint(0, self.diffusion.num_timesteps, (B,), device=self.device).long()

            # 传入原始长度的 obs_cond
            loss, model_out = self.diffusion.p_losses(
                t=t, x_0=x_0, obs=obs_cond, mask_past=mask_past, mask_all=masks_stages
            )
            
            output['loss'] = loss
            
            # 解析 model_out
            if model_out.dim() == 4: model_out = model_out[0]
            # model_out shape [B, n_query, C] (因为 slice 过了)
            
            # [保留修改] 反向映射 [-1, 1] -> [0, 1]
            pred_action_raw = model_out[:, :, :self.n_class]
            pred_action_probs = (pred_action_raw.clamp(-1, 1) + 1) / 2.0 
            pred_action_logits = torch.logit(pred_action_probs.clamp(min=1e-6, max=1-1e-6))
            
            pred_offset_raw = model_out[:, :, self.n_class]
            pred_offset_01 = (pred_offset_raw + 1) / 2.0
            pred_offset = pred_offset_01 * norm_factor 
            
            output['action'] = pred_action_logits
            output['offset'] = pred_offset
            
            if self.args.actionness:
                 pred_act_raw = model_out[:, :, self.n_class + 1]
                 pred_act_probs = (pred_act_raw.clamp(-1, 1) + 1) / 2.0
                 output['actionness'] = torch.logit(pred_act_probs.clamp(min=1e-6, max=1-1e-6))
            
        else:
            # --- 推理模式 ---
            sampled_x = self.diffusion.predict(
                x_0=torch.zeros((B, self.n_query, self.diff_out_dim), device=self.device),
                obs=obs_cond,
                mask_past=mask_past,
                masks_stages=masks_stages,
                n_samples=1,
                n_diffusion_steps=self.diffusion.ddim_timesteps
            )
            
            sampled_x = sampled_x[0] 
            
            # [保留修改] 反向映射
            pred_action_raw = sampled_x[:, :, :self.n_class]
            pred_action_probs = (pred_action_raw.clamp(-1, 1) + 1) / 2.0
            pred_action_logits = torch.logit(pred_action_probs.clamp(min=1e-6, max=1-1e-6))
            
            pred_offset_raw = sampled_x[:, :, self.n_class]
            pred_offset_01 = (pred_offset_raw + 1) / 2.0
            pred_offset = pred_offset_01 * norm_factor
            
            output['action'] = pred_action_logits
            output['offset'] = pred_offset
            
            if self.args.actionness:
                pred_act_raw = sampled_x[:, :, self.n_class + 1]
                pred_act_probs = (pred_act_raw.clamp(-1, 1) + 1) / 2.0
                output['actionness'] = torch.logit(pred_act_probs.clamp(min=1e-6, max=1-1e-6))

        return output

def get_pad_mask(seq, pad_idx):
    return (seq == pad_idx)
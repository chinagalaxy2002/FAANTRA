# model/futr.py
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

# [修改] 增加 offset_dim 和 actionness_dim 参数
class DiffusionConfig:
    def __init__(self, args, input_dim, num_classes, offset_dim=0, actionness_dim=0):
        self.layer_type = "mamba"
        self.kernel_size = 3
        self.num_stages = 1
        self.num_layers = args.num_encoder_layers if hasattr(args, 'num_encoder_layers') else 4
        self.model_dim = args.hidden_dim
        self.input_dim = input_dim
        self.num_classes = num_classes # 仅指动作类别
        self.offset_dim = offset_dim     # [新增]
        self.actionness_dim = actionness_dim # [新增]
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
        # [修改] diff_out_dim 只包含动作类别，不再累加 offset 和 actionness
        self.diff_out_dim = n_class 
        
        self.offset_dim = 1
        
        if args.actionness:
            self.actionness_dim = 1
        else:
            self.actionness_dim = 0

        # [修改] 传递分离的维度配置
        # 注意：这里的 input_dim 传给 Config 的是 hidden_dim (RegNet投影后的维度)
        diff_cfg = DiffusionConfig(args, hidden_dim, self.diff_out_dim, self.offset_dim, self.actionness_dim)
        self.denoise_model = BitDiffPredictorTCN(diff_cfg)

        sampling_steps = getattr(args, 'ddim_timesteps', 50) 
        
        self.diffusion = GaussianBitDiffusion(
            model=self.denoise_model,
            condition_x0=False,
            num_classes=self.diff_out_dim, # 这里仅指 Diffusion 需要处理的通道数 (即类别数)
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
        
        # 输入解析逻辑
        if isinstance(inputs, dict):
            src = inputs['obs']
            targets_dict = inputs
            if 'mode' in inputs:
                mode = inputs['mode']
        else:
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
        
        # --- 特征提取 ---
        # features: [B, S, InputDim]
        features = self.features(src.view(-1, C, H, W)).reshape(B, S, self.input_dim)
        # obs_feat: [B, S, HiddenDim]
        obs_feat = self.relu(self.input_proj(features))

        output = dict()
        if self.args.seg:
            output['seg'] = self.fc_seg(obs_feat)

        # Time-Concat 准备
        obs_cond = obs_feat 
        
        # 构造默认 Mask
        mask_past = torch.ones((B, S, 1), device=self.device)
        masks_stages = [torch.ones((B, self.n_query, 1), device=self.device)]
        
        norm_factor = float(self.args.clip_len)

        if mode == 'train':
            if targets_dict is None:
                raise ValueError("Diffusion Training requires targets_dict.")
            
            # 键名映射
            if 'action_target' in targets_dict: targets_dict['action'] = targets_dict['action_target']
            if 'offset_target' in targets_dict: targets_dict['offset'] = targets_dict['offset_target']
            if 'actionness_target' in targets_dict: targets_dict['actionness'] = targets_dict['actionness_target']

            # 随机采样时间步 t
            t = torch.randint(0, self.diffusion.num_timesteps, (B,), device=self.device).long()

            if 'x_0' in targets_dict and 'mask_past' in targets_dict:
                # [修改] 提取 Class Target
                x_0_cls = targets_dict['x_0']
                
                # 如果传入的 x_0 依然是拼接过的 (兼容旧 DataLoader)，我们需要拆分
                # 旧 x_0 维度可能是: n_class + 1 (offset) [+ 1 (actionness)]
                if x_0_cls.shape[-1] > self.n_class:
                     x_0_cls = x_0_cls[..., :self.n_class]
                
                # 映射到 [-1, 1]
                x_0 = x_0_cls * 2.0 - 1.0
                
                mask_past_for_diff = targets_dict['mask_past'] 
                masks_stages_for_diff = targets_dict['masks_stages']
                
                # [修改] 准备 Offset Target (映射到 [-1, 1])
                gt_offset = targets_dict['offset']
                gt_offset_norm = gt_offset / norm_factor
                offset_target = (gt_offset_norm.unsqueeze(-1) * 2.0) - 1.0
                
                # [修改] 准备 Actionness Target (映射到 [-1, 1])
                actionness_target = None
                if self.args.actionness and 'actionness' in targets_dict:
                    gt_act = targets_dict['actionness'].unsqueeze(-1)
                    actionness_target = gt_act * 2.0 - 1.0

                # [修改] 调用 Diffusion，传入分离的 Targets
                loss_dict = self.diffusion.p_losses(
                    t=t, 
                    x_0=x_0, # 仅 Class
                    obs=obs_cond, 
                    mask_past=mask_past_for_diff, 
                    mask_all=masks_stages_for_diff,
                    offset_target=offset_target,
                    actionness_target=actionness_target,
                    offset_loss_weight=getattr(self.args, 'offset_loss_weight', 1.0)
                )
                
                loss = loss_dict['loss']
                
                # 从 dict 中取回预测结果
                pred_cls = loss_dict['action']
                pred_off = loss_dict['offset']
                pred_act = loss_dict['actionness']

                # 反向映射 Class: [-1, 1] -> Logits
                pred_action_probs = (pred_cls.clamp(-1, 1) + 1) / 2.0
                output['action'] = torch.logit(pred_action_probs.clamp(min=1e-6, max=1-1e-6))
                
                # 反向映射 Offset: [-1, 1] -> [0, 1] -> 真实值
                if pred_off is not None:
                    pred_off_01 = (pred_off + 1) / 2.0
                    output['offset'] = pred_off_01 * norm_factor
                else:
                    output['offset'] = None
                
                # 反向映射 Actionness: [-1, 1] -> Logits
                if pred_act is not None:
                     pred_act_probs = (pred_act.clamp(-1, 1) + 1) / 2.0
                     output['actionness'] = torch.logit(pred_act_probs.clamp(min=1e-6, max=1-1e-6))
                else:
                     output['actionness'] = None

            else:
                # [Legacy 逻辑] 如果 DataLoader 没传 x_0，这里做个兜底 (虽然新代码通常都有)
                # 简单处理：报错或者手动构建
                raise NotImplementedError("New architecture requires 'x_0' in targets_dict pre-computed by DataLoader.")

            output['loss'] = loss
            
        else:
            # --- Inference ---
            if targets_dict and 'mask_past' in targets_dict:
                mask_past_infer = targets_dict['mask_past']
                masks_stages_infer = targets_dict['masks_stages']
            else:
                mask_past_infer = mask_past
                masks_stages_infer = masks_stages

            # [修改] x_0 初始化只针对 Class 维度
            x_0_infer = torch.zeros((B, self.n_query, self.diff_out_dim), device=self.device)

            # predict 返回的是拼接好的 [S, B, T, Total_Dim]
            sampled_x = self.diffusion.predict(
                x_0=x_0_infer,
                obs=obs_cond,
                mask_past=mask_past_infer,
                masks_stages=masks_stages_infer,
                n_samples=10, 
                n_diffusion_steps=self.diffusion.ddim_timesteps
            )
            
            # 先取平均 (Mean over samples) -> [B, T, Total_Dim]
            sampled_x_avg = sampled_x.mean(dim=0)
            # 此时已经是 [B, T, C]，不需要 rearrange 了 (除非 bit_diffusion 返回的是 [S, B, C, T])
            # 检查 bit_diffusion.py: 最后是 permute(0, 1, 3, 2) -> [S, B, C, T]
            # 所以 sampled_x_avg 是 [B, C, T]
            
            # 转回 [B, T, C] 以便切分
            sampled_x_avg = rearrange(sampled_x_avg, 'b c t -> b t c')

            # [修改] 切分输出
            # 顺序: Class -> Offset -> Actionness
            
            idx = 0
            # 1. Class
            pred_action_raw = sampled_x_avg[..., idx : idx + self.n_class]
            idx += self.n_class
            
            # 2. Offset
            pred_offset_raw = None
            if self.offset_dim > 0:
                pred_offset_raw = sampled_x_avg[..., idx : idx + self.offset_dim]
                idx += self.offset_dim
            
            # 3. Actionness
            pred_act_raw = None
            if self.actionness_dim > 0:
                pred_act_raw = sampled_x_avg[..., idx : idx + self.actionness_dim]

            # 反向映射逻辑
            pred_action_probs = (pred_action_raw.clamp(-1, 1) + 1) / 2.0
            output['action'] = torch.logit(pred_action_probs.clamp(min=1e-6, max=1-1e-6))
            
            if pred_offset_raw is not None:
                pred_offset_01 = (pred_offset_raw + 1) / 2.0
                output['offset'] = pred_offset_01 * norm_factor
            else:
                output['offset'] = None
            
            if pred_act_raw is not None:
                pred_act_probs = (pred_act_raw.clamp(-1, 1) + 1) / 2.0
                output['actionness'] = torch.logit(pred_act_probs.clamp(min=1e-6, max=1-1e-6))
            else:
                output['actionness'] = None

        return output
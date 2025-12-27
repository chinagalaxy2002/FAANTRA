# 导入必要的库
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import math
import os
import sys
import pdb  # Python调试器
import torchvision.transforms as T  # 图像变换库
from einops import repeat, rearrange  # 张量操作库，用于灵活的维度重排
from model.extras.transformer import Transformer  # 自定义Transformer模块
from model.extras.position import PositionalEncoding  # 位置编码模块
import timm  # PyTorch图像模型库，提供预训练模型
from model.T_Deed_Modules.modules import EDSGPMIXERLayers  # 时序特征增强模块
from model.T_Deed_Modules.shift import make_temporal_shift  # 时序偏移模块（TSM）

# 添加父目录到系统路径
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

# FUTR主模型类：Football Action Anticipation Transformer
class FUTR(nn.Module):

    def __init__(self, n_class, hidden_dim, src_pad_idx, device, args, n_query=8, n_head=8,
                 num_encoder_layers=6, num_decoder_layers=6, src_attn_mask=None, tgt_attn_mask=None):
        """
        参数说明:
            n_class: 动作类别数（包括背景类）
            hidden_dim: Transformer隐藏层维度
            src_pad_idx: 源序列的填充索引
            device: 计算设备 (CPU/GPU)
            args: 命令行参数对象
            n_query: 查询数量（预测未来多少个动作）
            n_head: 多头注意力的头数
            num_encoder_layers: Transformer编码器层数
            num_decoder_layers: Transformer解码器层数
            src_attn_mask: 源序列注意力掩码
            tgt_attn_mask: 目标序列注意力掩码
        """
        super().__init__()
        
        # 检查参数合法性：如果没有解码器，query数量必须为1
        if num_decoder_layers < 1 and n_query > 1:
            raise ValueError(f"n_query must be 1 if no decoder is to be used\nGiven values are: {n_query} and {num_decoder_layers} respectively")
        
        # 保存基本配置
        self.encoder_only = num_decoder_layers == 0  # 是否只使用编码器
        self.src_pad_idx = src_pad_idx  # 填充索引
        self.device = device
        self.hidden_dim = hidden_dim
        self.feature_arch = args.feature_arch  # 特征提取架构名称
        self.temp_arch = args.temporal_arch  # 时序建模架构名称
        self.src_attn_mask = src_attn_mask  # 源注意力掩码
        self.tgt_attn_mask = tgt_attn_mask  # 目标注意力掩码
        self.jointtrain_available = args.jointtrain is not None  # 是否进行联合训练（多数据集）
        
        # ========== 特征提取器设置 ==========
        # 使用RegNetY系列作为空间特征提取backbone
        if self.feature_arch.startswith(('rny002', 'rny004', 'rny006', 'rny008')):
            # 创建timm预训练模型
            self.features = timm.create_model({
                'rny002': 'regnety_002',  # 最小模型
                'rny004': 'regnety_004',
                'rny006': 'regnety_006',
                'rny008': 'regnety_008',  # 最大模型
            }[self.feature_arch.rsplit('_', 1)[0]], pretrained=True)  # 移除后缀后查找对应模型
            
            # 获取特征维度（最后一层的输入维度）
            feat_dim = self.features.head.fc.in_features

            # 移除分类头，只保留特征提取部分
            self.features.head.fc = nn.Identity()
            self.input_dim = feat_dim
        else:
            raise NotImplementedError(args.feature_arch)
        
        # ========== 添加时序偏移模块 (Temporal Shift Module, TSM) ==========
        # TSM可以在不增加计算量的情况下，让2D CNN捕获时序信息
        # 计算最大观察长度
        # 如果使用cheating数据集（可以看到未来帧），使用cheating_range
        # 否则使用obs_perc（观察百分比）的最大值
        max_obs_len = int(args.clip_len*args.cheating_range[1])-int(args.clip_len*args.cheating_range[0]) if args.cheating_dataset else int(args.clip_len*max(args.obs_perc))
        
        if self.feature_arch.endswith('_gsm'):  # Global Shift Module
            make_temporal_shift(self.features, max_obs_len, mode='gsm')
        elif self.feature_arch.endswith('_gsf'):  # Global Shift Fusion
            make_temporal_shift(self.features, max_obs_len, mode='gsf')

        # ========== 时序架构设置 ==========
        if self.temp_arch == 'ed_sgp_mixer':
            # ED-SGP-MIXER: Efficient and Deep Spatial-Grouped Pooling Mixer
            # 可学习的时序位置编码（不同于标准的sinusoidal编码）
            self.temp_enc = nn.Parameter(torch.normal(mean=0, std=1/max_obs_len, size=(max_obs_len, self.input_dim)))
            
            # SGP Mixer层：用于时序特征增强
            self.temp_fine = EDSGPMIXERLayers(
                self.input_dim,  # 输入维度
                max_obs_len,     # 序列长度
                num_layers=args.n_layers,  # 层数
                ks=args.sgp_ks,  # 卷积核大小
                k=args.sgp_r,    # 分组数
                concat=True      # 是否连接残差
            )

        # ========== 特征嵌入层 ==========
        # 将特征维度映射到Transformer的隐藏维度
        self.input_embed = nn.Linear(self.input_dim, hidden_dim)
        
        # ========== Transformer核心模块 ==========
        self.transformer = Transformer(
            hidden_dim,           # 隐藏维度
            n_head,               # 注意力头数
            num_encoder_layers,   # 编码器层数
            num_decoder_layers,   # 解码器层数
            hidden_dim*4,         # FFN维度（通常是hidden_dim的4倍）
            normalize_before=False  # 是否在注意力前进行LayerNorm（Post-LN）
        )
        
        # ========== Query Embedding ==========
        # 可学习的动作查询嵌入：每个query对应一个未来的预测动作
        self.n_query = n_query
        self.args = args
        nn.init.xavier_uniform_(self.input_embed.weight)  # Xavier初始化
        self.query_embed = nn.Embedding(self.n_query, hidden_dim)


        # ========== 任务头1: 动作分割 (Action Segmentation) ==========
        if args.seg:
            # 分割头：预测观察帧中每一帧的动作类别
            self.fc_seg = nn.Linear(hidden_dim, n_class)
            nn.init.xavier_uniform_(self.fc_seg.weight)
            
            # 联合训练的分割头（用于第二个数据集）
            if self.jointtrain_available:
                # +1是为了给第二个数据集添加背景类
                self.fc_seg_jointtrain = nn.Linear(hidden_dim, args.jointtrain['num_classes'] + 1)
                nn.init.xavier_uniform_(self.fc_seg_jointtrain.weight)

        # ========== 任务头2: 动作预期 (Action Anticipation) ==========
        if args.anticipate:
            # 动作分类头：预测未来动作的类别
            # 注意：如果使用actionness，则不需要预测背景类（-1*args.actionness）
            # 因为actionness会单独预测某个位置是否有动作
            self.fc = nn.Linear(hidden_dim, n_class - 1*args.actionness)
            nn.init.xavier_uniform_(self.fc.weight)
            
            # 时间偏移头：预测每个动作距离当前的时间偏移
            self.fc_len = nn.Linear(hidden_dim, 1)
            nn.init.xavier_uniform_(self.fc_len.weight)
            
            # 联合训练的预期头
            if self.jointtrain_available:
                self.fc_jointtrain = nn.Linear(hidden_dim, args.jointtrain['num_classes'] + 1 - 1*args.actionness)
                nn.init.xavier_uniform_(self.fc_jointtrain.weight)
                self.fc_len_jointtrain = nn.Linear(hidden_dim, 1)
                nn.init.xavier_uniform_(self.fc_len_jointtrain.weight)
        
        # ========== 任务头3: 动作性 (Actionness) ==========
        if args.actionness:
            # 动作性头：预测每个query位置是否真的存在动作（二分类）
            # 用于区分真实动作和"无动作"的query
            self.fc_actionness = nn.Linear(hidden_dim, 1)
            nn.init.xavier_uniform_(self.fc_actionness.weight)
            
            # 联合训练的动作性头
            if self.jointtrain_available:
                self.fc_actionness_jointtrain = nn.Linear(hidden_dim, 1)
                nn.init.xavier_uniform_(self.fc_actionness_jointtrain.weight)

        # ========== 位置编码 ==========
        if args.pos_emb:
            # 可学习的位置嵌入
            max_seq_len = args.max_pos_len  # 最大序列长度
            self.pos_embedding = nn.Parameter(torch.zeros(1, max_seq_len, hidden_dim))
            nn.init.xavier_uniform_(self.pos_embedding)
            
            # Sinusoidal位置编码（标准Transformer风格）
            self.pos_enc = PositionalEncoding(hidden_dim)

        # ========== 数据预处理 ==========
        # 数据增强：只在训练时使用
        self.augmentation = T.Compose([
            T.RandomApply([T.ColorJitter(hue=0.2)], p=0.25),              # 色调变换
            T.RandomApply([T.ColorJitter(saturation=(0.7, 1.2))], p=0.25), # 饱和度变换
            T.RandomApply([T.ColorJitter(brightness=(0.7, 1.2))], p=0.25), # 亮度变换
            T.RandomApply([T.ColorJitter(contrast=(0.7, 1.2))], p=0.25),   # 对比度变换
            T.RandomApply([T.GaussianBlur(5)], p=0.25),                    # 高斯模糊
            T.RandomHorizontalFlip(),                                       # 水平翻转
        ])
        
        # 标准化：使用ImageNet的均值和标准差
        self.standarization = T.Compose([
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])
        
    def augment(self, x):
        """对batch中的每个样本应用数据增强"""
        for i in range(x.shape[0]):
            x[i] = self.augmentation(x[i])
        return x

    def standarize(self, x):
        """对batch中的每个样本应用标准化"""
        for i in range(x.shape[0]):
            x[i] = self.standarization(x[i])
        return x

    # 前向传播函数
    def forward(self, inputs, mode='train'):
        """
        参数:
            inputs: 训练模式下是(src, src_label)，测试模式下只有src
                   src的形状: [Batch, Sequence, Channel, Height, Width]
            mode: 'train' 或 'test'
        
        返回:
            output: 字典，包含不同任务头的输出
        """
        
        # ========== 准备输入和掩码 ==========
        if mode == 'train':
            src, src_label = inputs  # 训练时有标签
            tgt_key_padding_mask = None
            
            # 创建源序列的padding掩码（标记哪些位置是填充的）
            src_key_padding_mask = get_pad_mask(src_label, self.src_pad_idx).to(self.device)
            
            # memory掩码用于解码器关注编码器输出时使用
            memory_key_padding_mask = src_key_padding_mask.clone().to(self.device)
        else:  # 测试模式
            src = inputs
            src_key_padding_mask = None
            memory_key_padding_mask = None
            tgt_key_padding_mask = None

        # 注意力掩码（用于控制哪些位置可以互相关注）
        src_mask = self.src_attn_mask  # 源序列自注意力掩码
        tgt_mask = self.tgt_attn_mask  # 目标序列自注意力掩码（通常是因果掩码）

        # ========== 步骤1: 空间特征提取 ==========
        B, S, C, H, W = src.size()  # Batch, Sequence, Channel, Height, Width
        
        # 归一化到[0, 1]范围
        src = src / 255.0
        
        # 训练时应用数据增强
        if mode == "train":
            src = self.augment(src)
            print(f"数据增强后的形状 (src): {src.shape}")
        
        # 应用ImageNet标准化
        src = self.standarize(src)
        
        # 通过CNN backbone提取空间特征
        # 先将(B, S, C, H, W)重塑为(B*S, C, H, W)，批量处理所有帧
        # 然后重塑回(B, S, Feature_Dim)
        src = self.features(src.view(-1, C, H, W)).reshape(B, S, self.input_dim)
        
        print("--- 步骤 1: 空间特征提取完成 ---")
        print(f"输出特征序列的形状 (Batch, Sequence, Feature_Dim): {src.shape}")
        print("--------------------------------------")
        
        # ========== 步骤2: 时序特征增强 (可选) ==========
        if self.temp_arch == 'ed_sgp_mixer':
            # 添加可学习的时序位置编码
            src = src + self.temp_enc.expand(B, -1, -1)
            
            # 通过SGP Mixer增强时序特征
            src = self.temp_fine(src)
            print(f"时序增强后的特征序列 (src): {src.shape}")
        
        # ========== 步骤3: 特征嵌入 ==========
        # 将特征维度映射到Transformer的隐藏维度
        src = self.input_embed(src)  # [B, S, hidden_dim]
        print(f"嵌入后的特征序列 (src): {src.shape}")
        
        # 应用ReLU激活
        src = F.relu(src)
        print(f"ReLU激活后 (src): {src.shape}")
        
        # ========== 步骤4: 准备Query Embedding ==========
        # 获取可学习的动作查询嵌入
        action_query = self.query_embed.weight  # [n_query, hidden_dim]
        
        # 复制到batch中的每个样本
        action_query = action_query.unsqueeze(0).repeat(B, 1, 1)  # [B, n_query, hidden_dim]
        
        # 初始化目标序列为零（Transformer解码器的输入）
        tgt = torch.zeros_like(action_query)

        # ========== 步骤5: 位置编码 ==========
        if self.encoder_only:
            # 只用编码器模式：需要为源序列+query添加位置编码
            pos = self.pos_embedding[:, :S+1,].repeat(B, 1, 1)
            
            # 调整padding掩码，为额外的query位置添加False（不mask）
            if src_key_padding_mask is not None:
                false_append = torch.tensor([False], device=src_key_padding_mask.device).expand((src_key_padding_mask.shape[0], 1))
                src_key_padding_mask = torch.cat((src_key_padding_mask, false_append), dim=1)
        else:
            # 编码器-解码器模式：只为源序列添加位置编码
            pos = self.pos_embedding[:, :S,].repeat(B, 1, 1)
        
        # ========== 步骤6: 重排维度 ==========
        # Transformer期望的输入格式是 [Sequence, Batch, Feature]
        src = rearrange(src, 'b t c -> t b c')  # [S, B, C]
        tgt = rearrange(tgt, 'b t c -> t b c')  # [n_query, B, C]
        pos = rearrange(pos, 'b t c -> t b c')  # [S, B, C] 或 [S+1, B, C]
        action_query = rearrange(action_query, 'b t c -> t b c')  # [n_query, B, C]
        
        # ========== 步骤7: Transformer处理 ==========
        # 输入:
        #   - src: 编码器输入（观察帧的特征）
        #   - tgt: 解码器输入（初始化为零）
        #   - src_key_padding_mask: 源序列的padding掩码
        #   - src_mask: 源序列的注意力掩码
        #   - tgt_mask: 目标序列的注意力掩码（因果掩码）
        #   - action_query: 可学习的query嵌入
        #   - pos: 位置编码
        # 输出:
        #   - src: 编码器输出（用于分割任务）
        #   - tgt: 解码器输出（用于预期任务）
        src, tgt = self.transformer(
            src, tgt, 
            src_key_padding_mask, src_mask, tgt_mask, 
            None,  # memory_mask
            action_query, pos, None  # query_pos
        )

        # ========== 步骤8: 恢复维度 ==========
        # 将Transformer输出从 [Sequence, Batch, Feature] 转回 [Batch, Sequence, Feature]
        tgt = rearrange(tgt, 't b c -> b t c')  # [B, n_query, hidden_dim]
        src = rearrange(src, 't b c -> b t c')  # [B, S, hidden_dim]

        # ========== 步骤9: 多任务头输出 ==========
        output = dict()
        
        # 任务1: 动作预期 (Action Anticipation)
        if self.args.anticipate:
            # 动作分类：预测每个query的动作类别
            output_class = self.fc(tgt)  # [B, n_query, n_class-1] (如果用actionness)
            
            # 时间偏移：预测每个动作的时间位置
            offset = self.fc_len(tgt)  # [B, n_query, 1]
            offset = offset.squeeze(2)  # [B, n_query]
            
            # 联合训练：拼接两个数据集的预测
            if self.jointtrain_available:
                output_class_jointtrain = self.fc_jointtrain(tgt)
                offset_jointtrain = self.fc_len_jointtrain(tgt)
                offset_jointtrain = offset_jointtrain.squeeze(2)
                
                # 在类别维度上拼接
                output_class = torch.cat([output_class, output_class_jointtrain], dim=2)
                # 在query维度上拼接
                offset = torch.cat([offset, offset_jointtrain], dim=1)
            
            output['offset'] = offset      # 时间偏移
            output['action'] = output_class  # 动作类别
            print(f"偏移形状: {offset.shape}")
            print(f"动作分类形状: {output_class.shape}")
        
        # 任务2: 动作分割 (Action Segmentation)
        if self.args.seg:
            # 分割：为观察帧中的每一帧预测动作类别
            tgt_seg = self.fc_seg(src)  # [B, S, n_class]
            
            # 联合训练
            if self.jointtrain_available:
                tgt_seg_jointtrain = self.fc_seg_jointtrain(src)
                tgt_seg = torch.cat([tgt_seg, tgt_seg_jointtrain], dim=2)
            
            output['seg'] = tgt_seg
            print(f"分割形状: {tgt_seg.shape}")
        
        # 任务3: 动作性 (Actionness)
        if self.args.actionness:
            # 动作性：预测每个query是否真的对应一个动作
            actionness = self.fc_actionness(tgt)  # [B, n_query, 1]
            actionness = actionness.squeeze(2)     # [B, n_query]
            
            # 联合训练
            if self.jointtrain_available:
                actionness_jointtrain = self.fc_actionness_jointtrain(tgt)
                actionness_jointtrain = actionness_jointtrain.squeeze(2)
                actionness = torch.cat([actionness, actionness_jointtrain], dim=1)
            
            output['actionness'] = actionness
            print(f"动作性形状: {actionness.shape}")
        
        return output


# 工具函数：生成padding掩码
def get_pad_mask(seq, pad_idx):
    """
    生成padding掩码矩阵
    
    参数:
        seq: 序列标签 [B, S]
        pad_idx: 填充值的索引
    
    返回:
        掩码矩阵 [B, S]，True表示该位置是padding
    """
    return (seq == pad_idx)
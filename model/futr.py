import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import math
import os
import sys
import pdb
import torchvision.transforms as T
from einops import repeat, rearrange
from model.extras.transformer import Transformer
from model.extras.position import PositionalEncoding
import timm
from model.T_Deed_Modules.modules import UniFormerNeck
from model.T_Deed_Modules.shift import make_temporal_shift

# Try importing transformers for VideoMAE
try:
    from transformers import VideoMAEModel, VideoMAEConfig
except ImportError:
    print("Warning: transformers library not installed. VideoMAE will fail.")

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

class FUTR(nn.Module):

    def __init__(self, n_class, hidden_dim, src_pad_idx, device, args, n_query=8, n_head=8,
                 num_encoder_layers=6, num_decoder_layers=6, src_attn_mask=None, tgt_attn_mask=None):
        super().__init__()
        if num_decoder_layers < 1 and n_query > 1:
            raise ValueError(f"n_query must be 1 if no decoder is to be used\nGiven values are: {n_query} and {num_decoder_layers} respectively")
        self.encoder_only = num_decoder_layers == 0
        self.src_pad_idx = src_pad_idx
        self.device = device
        self.hidden_dim = hidden_dim
        self.feature_arch = args.feature_arch
        self.temp_arch = args.temporal_arch
        self.src_attn_mask = src_attn_mask
        self.tgt_attn_mask = tgt_attn_mask
        self.jointtrain_available = args.jointtrain is not None
        
        # --- 改进点 1: 特征提取网络 (Backbone) ---
        if 'videomae' in self.feature_arch:
            print(f"Loading VideoMAE Backbone: {self.feature_arch}")
            # Example: feature_arch string could be 'videomae_base' or 'videomae_large'
            # Defaulting to a standard hub path. User can change this map.
            model_map = {
                'videomae_base': 'MCG-NJU/videomae-base',
                'videomae_large': 'MCG-NJU/videomae-large',
                'videomae_huge': 'MCG-NJU/videomae-huge-finetuned-kinetics', # Example for huge
            }
            model_name = model_map.get(self.feature_arch, 'MCG-NJU/videomae-base')
            
            # Load VideoMAE from HuggingFace
            self.features = VideoMAEModel.from_pretrained(model_name)
            self.input_dim = self.features.config.hidden_size
            self.is_video_transformer = True
            
        elif self.feature_arch.startswith(('rny002', 'rny004', 'rny006', 'rny008')):
            self.features = timm.create_model({
                'rny002': 'regnety_002',
                'rny004': 'regnety_004',
                'rny006': 'regnety_006',
                'rny008': 'regnety_008',
            }[self.feature_arch.rsplit('_', 1)[0]], pretrained=True)
            feat_dim = self.features.head.fc.in_features
            # Remove final classification layer
            self.features.head.fc = nn.Identity()
            self.input_dim = feat_dim
            self.is_video_transformer = False
        else:
            raise NotImplementedError(args.feature_arch)
        
        # Add Temporal Shift Modules (Only for CNNs usually, but kept logic compatible)
        # NOTE: NEED TO CHANGE 2ND ARGUMENT FOR CHEATING DATASET
        max_obs_len = int(args.clip_len*args.cheating_range[1])-int(args.clip_len*args.cheating_range[0]) if args.cheating_dataset else int(args.clip_len*max(args.obs_perc))
        
        if not self.is_video_transformer and self.feature_arch.endswith('_gsm'):
            make_temporal_shift(self.features, max_obs_len, mode='gsm')
        elif not self.is_video_transformer and self.feature_arch.endswith('_gsf'):
            make_temporal_shift(self.features, max_obs_len, mode='gsf')

        # --- 改进点 2: 颈部网络 (Neck) - UniFormer ---
        # 替换原有的 EDSGPMIXERLayers 为 UniFormerNeck
        if self.temp_arch == 'uniformer' or self.temp_arch == 'ed_sgp_mixer': # Overwrite existing flag or use new one
            print("Using UniFormer Neck for Temporal Interaction")
            self.temp_enc = nn.Parameter(torch.normal(mean = 0, std = 1 / max_obs_len, size = (max_obs_len, self.input_dim)))
            # Using the new UniFormerNeck defined in modules.py
            self.temp_fine = UniFormerNeck(self.input_dim, max_obs_len, num_layers=args.n_layers)
        else:
            self.temp_fine = None

        self.input_embed = nn.Linear(self.input_dim, hidden_dim)
        self.transformer = Transformer(hidden_dim, n_head, num_encoder_layers, num_decoder_layers,
                                        hidden_dim*4, normalize_before=False)
        self.n_query = n_query
        self.args = args
        nn.init.xavier_uniform_(self.input_embed.weight)
        self.query_embed = nn.Embedding(self.n_query, hidden_dim)

        if args.seg :
            self.fc_seg = nn.Linear(hidden_dim, n_class)
            nn.init.xavier_uniform_(self.fc_seg.weight)
            if self.jointtrain_available:
                self.fc_seg_jointtrain = nn.Linear(hidden_dim, args.jointtrain['num_classes'] + 1)  # +1 for background class
                nn.init.xavier_uniform_(self.fc_seg_jointtrain.weight)

        if args.anticipate :
            # Anticipation head has the capacity to predict background class despite it not being anywhere in anticipation
            # To avoid this I will make the EOS token the same number as the background class
            self.fc = nn.Linear(hidden_dim, n_class - 1*args.actionness)
            nn.init.xavier_uniform_(self.fc.weight)
            self.fc_len = nn.Linear(hidden_dim, 1)
            nn.init.xavier_uniform_(self.fc_len.weight)
            if self.jointtrain_available:
                self.fc_jointtrain = nn.Linear(hidden_dim, args.jointtrain['num_classes'] + 1 - 1*args.actionness)
                nn.init.xavier_uniform_(self.fc_jointtrain.weight)
                self.fc_len_jointtrain = nn.Linear(hidden_dim, 1)
                nn.init.xavier_uniform_(self.fc_len_jointtrain.weight)
        
        if args.actionness :
            self.fc_actionness = nn.Linear(hidden_dim, 1)
            nn.init.xavier_uniform_(self.fc_actionness.weight)
            if self.jointtrain_available:
                self.fc_actionness_jointtrain = nn.Linear(hidden_dim, 1)
                nn.init.xavier_uniform_(self.fc_actionness_jointtrain.weight)

        if args.pos_emb:
            #pos embedding
            max_seq_len = args.max_pos_len
            self.pos_embedding = nn.Parameter(torch.zeros(1, max_seq_len, hidden_dim))
            nn.init.xavier_uniform_(self.pos_embedding)
            # Sinusoidal position encoding
            self.pos_enc = PositionalEncoding(hidden_dim)

        # Preprocessing
        # Augmentations
        self.augmentation = T.Compose([
            T.RandomApply([T.ColorJitter(hue = 0.2)], p = 0.25),
            T.RandomApply([T.ColorJitter(saturation = (0.7, 1.2))], p = 0.25),
            T.RandomApply([T.ColorJitter(brightness = (0.7, 1.2))], p = 0.25),
            T.RandomApply([T.ColorJitter(contrast = (0.7, 1.2))], p = 0.25),
            T.RandomApply([T.GaussianBlur(5)], p = 0.25),
            T.RandomHorizontalFlip(),
        ])
        #Standarization
        self.standarization = T.Compose([
            T.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225)) #Imagenet mean and std
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
        if mode == 'train' :
            src, src_label = inputs
            tgt_key_padding_mask = None
            src_key_padding_mask = get_pad_mask(src_label, self.src_pad_idx).to(self.device)
            memory_key_padding_mask = src_key_padding_mask.clone().to(self.device)
        else :
            src = inputs
            src_key_padding_mask = None
            memory_key_padding_mask = None
            tgt_key_padding_mask = None

        src_mask = self.src_attn_mask
        tgt_mask = self.tgt_attn_mask

        B, S, C, H, W = src.size()
        src = src/255.0         # Normalize
        if mode == "train":
            src = self.augment(src) #augmentation per-batch
        src = self.standarize(src) #standarization imagenet stats

        # --- Backbone Forward Pass ---
        if self.is_video_transformer:
            # VideoMAE expects [B, T, C, H, W]
            # It processes the whole clip at once usually.
            # If S is large, this might be heavy. Assuming standard clip lengths (~16-64 frames).
            # VideoMAE outputs last_hidden_state: [B, T_patches, Hidden] or [B, Hidden] depending on config.
            # We want token-level features for each frame if possible, or pooled features per frame.
            
            # Note: Standard VideoMAE treats time as a dimension in patches.
            # Output is (batch_size, sequence_length, hidden_size).
            # To adapt for this pipeline which expects [B, S, Input_Dim], we feed (B, S, C, H, W).
            
            # HuggingFace VideoMAE forward:
            # pixel_values: (batch_size, num_frames, num_channels, height, width)
            video_outputs = self.features(pixel_values=src)
            src = video_outputs.last_hidden_state 
            
            # VideoMAE output shape is [B, Patch_Tokens, Hidden].
            # This is NOT [B, S, Hidden]. It collapses Time and Space.
            # For simplicity in this adaptation, we can Mean Pool the spatial patches to get Temporal features
            # Or use the CLS token if it exists (VideoMAE usually uses mean pooling).
            
            # IMPORTANT: VideoMAE V2 / standard VideoMAE collapses T/H/W into tokens.
            # We need to map back to S (frames) if we want frame-wise temporal modeling in the Transformer.
            # However, since VideoMAE already does spatiotemporal modeling, we can treat the output tokens 
            # as the "sequence" for the subsequent Transformer.
            # But the existing pipeline expects `S` frames. 
            
            # Strategy: Average pool spatial tokens to get 1 token per frame (approximate).
            # VideoMAE patch embedding usually has tubelet_size (t, h, w).
            # Assuming tubelet_t = 2 (standard), resulting T' = S / 2.
            # For implementation safety, we will just use the output sequence as 'src'.
            # B, Tokens, Dim = src.shape
            pass
            
        else:
            # CNN Approach (RegNet)
            src = self.features(src.view(-1, C, H, W)).reshape(B, S, self.input_dim)

        # --- Neck Forward Pass ---
        if self.temp_fine is not None: # Using UniFormerNeck
            if not self.is_video_transformer:
                src = src + self.temp_enc.expand(B, -1, -1)
            src = self.temp_fine(src)
            
        src = self.input_embed(src) #[B, S, C]
        src = F.relu(src)

        # action query embedding
        action_query = self.query_embed.weight
        action_query = action_query.unsqueeze(0).repeat(B, 1, 1)
        tgt = torch.zeros_like(action_query)

        # pos embedding
        if self.encoder_only:
            # Adjust pos embedding length if backbone changed sequence length (e.g. VideoMAE downsampling)
            curr_seq_len = src.shape[1]
            pos = self.pos_embedding[:, :curr_seq_len+1,].repeat(B, 1, 1)
            
            if src_key_padding_mask is not None:
                # Resize padding mask if sequence length changed
                if src_key_padding_mask.shape[1] != curr_seq_len:
                    # Simple interpolation for mask (nearest)
                    src_key_padding_mask = F.interpolate(src_key_padding_mask.unsqueeze(1).float(), size=curr_seq_len, mode='nearest').squeeze(1).bool()

                false_append = torch.tensor([False], device=src_key_padding_mask.device).expand((src_key_padding_mask.shape[0], 1))
                src_key_padding_mask = torch.cat((src_key_padding_mask, false_append),dim=1)
        else:
            curr_seq_len = src.shape[1]
            pos = self.pos_embedding[:, :curr_seq_len,].repeat(B, 1, 1)
            
        src = rearrange(src, 'b t c -> t b c')
        tgt = rearrange(tgt, 'b t c -> t b c')
        pos = rearrange(pos, 'b t c -> t b c')
        action_query = rearrange(action_query, 'b t c -> t b c')
        src, tgt = self.transformer(src, tgt, src_key_padding_mask, src_mask, tgt_mask, None, action_query, pos, None)

        tgt = rearrange(tgt, 't b c -> b t c')
        src = rearrange(src, 't b c -> b t c')

        output = dict()
        
        # --- 改进点 3: 保存 Embedding 用于对比学习 ---
        # 我们保存经过 Decoder 后的 tgt 特征，这些特征包含了对未来动作的上下文预测
        output['embedding'] = tgt # [B, n_query, hidden_dim]

        if self.args.anticipate :
            # action anticipation
            output_class = self.fc(tgt) #[B, T, C]
            offset = self.fc_len(tgt) #[B, T, 1]
            offset = offset.squeeze(2) #[B, T]
            if self.jointtrain_available:
                output_class_jointtrain = self.fc_jointtrain(tgt)
                offset_jointtrain = self.fc_len_jointtrain(tgt)
                offset_jointtrain = offset_jointtrain.squeeze(2)
                output_class = torch.cat([output_class, output_class_jointtrain], dim = 2)
                offset = torch.cat([offset, offset_jointtrain], dim = 1)
            output['offset'] = offset
            output['action'] = output_class

        if self.args.seg :
            # action segmentation
            tgt_seg = self.fc_seg(src)
            if self.jointtrain_available:
                tgt_seg_jointtrain = self.fc_seg_jointtrain(src)
                tgt_seg = torch.cat([tgt_seg, tgt_seg_jointtrain], dim = 2)
            output['seg'] = tgt_seg
        
        if self.args.actionness :
            # actionness
            actionness = self.fc_actionness(tgt)    #[B, T, 1]
            actionness = actionness.squeeze(2)      #[B, T]
            if self.jointtrain_available:
                actionness_jointtrain = self.fc_actionness_jointtrain(tgt)
                actionness_jointtrain = actionness_jointtrain.squeeze(2)      #[B, T]
                actionness = torch.cat([actionness, actionness_jointtrain], dim = 1)
            output['actionness'] = actionness

        return output


def get_pad_mask(seq, pad_idx):
    return (seq ==pad_idx)
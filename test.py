# 导入必要的库
import torch           # PyTorch深度学习框架
import json           # 用于处理JSON配置文件
import os             # 操作系统相关功能，如文件路径操作
import argparse       # 命令行参数解析
import numpy as np    # 数值计算库
from opts import update_args        # 从opts.py导入参数更新函数
from util.dataset import load_classes  # 从util.dataset导入类别加载函数
from torch import nn               # PyTorch神经网络模块
from eval import evaluate          # 导入通用评估函数
from eval_BAA import evaluate_BAA  # 导入Ball Action Anticipation数据集专用评估函数
from model.futr import FUTR        # 导入FUTR模型（Future Transformer）

def main():
    # 创建命令行参数解析器
    args = argparse.ArgumentParser()
    
    # 添加必需的命令行参数
    args.add_argument("config", type=str, help="Path to config file")      # 配置文件路径
    args.add_argument('checkpoint', type=str, help='Path to checkpoint')   # 模型检查点文件路径
    args.add_argument('model', type=str, help='Model name')                # 模型名称
    
    # 添加可选的命令行参数
    args.add_argument('-s', '--split', type=str, default="test", choices=["train", "val", "test", "challenge"],
                        help='Split to test on.')  # 选择要测试的数据集分割（训练/验证/测试/挑战集）
    args.add_argument('-o', '--overlap', type=float, default=0.5,
                        help='Overlap between clips (0.5 is no overlap between observations). Only applicable to BAS dataset')  # 视频片段之间的重叠率，仅适用于BAS数据集
    
    # 来自FUTR模型的参数
    args.add_argument("--cpu", action='store_true', help='run in cpu')  # 强制使用CPU运行
    
    # 解析命令行参数
    args = args.parse_args()
    
    # 加载JSON配置文件
    config_path = args.config
    with open(config_path, "r") as f:
        config = json.load(f)
    
    # 使用配置文件中的参数更新命令行参数
    args = update_args(args, config)
    
    # 设置填充索引，用于处理变长序列
    pad_idx = 255
    
    # 从class.txt文件加载动作类别字典
    actions_dict = load_classes(os.path.join('data', args.dataset, 'class.txt'))
    
    # 计算有效类别数量（总类别数减去要排除的类别数）
    n_class = len(actions_dict) - len(args.excluded_classes)
    
    # 根据参数选择计算设备（CPU或GPU）
    if args.cpu:
        device = torch.device('cpu')
        print('using cpu')
    else:
        device = torch.device('cuda')
        print('using gpu')
    
    # 加载预训练模型检查点
    #checkpoint = torch.load(args.checkpoint, map_location=device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    # 打印配置信息
    print("Config path:", config_path)
    print("Checkpoint path:", args.checkpoint)
    print("Overlap:", args.overlap)
    
    # 模型规格设置 - 初始化注意力掩码
    src_attn_mask = None  # 源序列（编码器）注意力掩码
    tgt_attn_mask = None  # 目标序列（解码器）注意力掩码
    
    # 如果启用了注意力掩码
    if args.mask_attn:
        # 计算观察序列的最大长度
        max_obs_len = int(args.clip_len*max(args.obs_perc))
        
        # 创建源序列注意力掩码（初始全部为True，即屏蔽所有位置）
        src_attn_mask = torch.full((max_obs_len, max_obs_len), True).to(device)
        # 创建目标序列注意力掩码
        tgt_attn_mask = torch.full((args.n_query, args.n_query), True).to(device)
        
        # 为源序列设置注意力窗口（只允许窗口内的位置相互注意）
        for i in range(max_obs_len):
            start = max(0, i - (args.mask_attn_window_src//2))      # 窗口开始位置
            end = min(max_obs_len, i + (args.mask_attn_window_src//2) + 1)  # 窗口结束位置
            src_attn_mask[i, start:end] = False  # False表示允许注意，True表示屏蔽
        
        # 为目标序列设置注意力窗口
        for i in range(args.n_query):
            start = max(0, i - (args.mask_attn_window_tgt//2))
            end = min(args.n_query, i + (args.mask_attn_window_tgt//2) + 1)
            tgt_attn_mask[i, start:end] = False
    
    # 初始化FUTR模型
    model = FUTR(n_class, args.hidden_dim, device=device, args=args, src_pad_idx=pad_idx,
                            n_query=args.n_query, n_head=args.n_head,
                            num_encoder_layers=args.n_encoder_layer, num_decoder_layers=args.n_decoder_layer,
                            src_attn_mask=src_attn_mask, tgt_attn_mask=tgt_attn_mask).to(device)
    
    # 使用DataParallel进行多GPU并行计算
    model = nn.DataParallel(model).to(device)
    
    # 检查检查点类型并加载模型权重
    if "model_state_dict" in checkpoint.keys():
        # 如果是训练检查点（包含完整训练状态）
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        # 如果只是模型权重
        model.load_state_dict(checkpoint)
    
    # 评估模型并打印结果
    if args.split == "challenge":
        # 挑战集评估（通常不返回结果，只生成预测文件）
        if args.dataset == 'soccernetballanticipation':
            # 使用BAA数据集专用评估函数
            evaluate_BAA(args.split, model, n_class, actions_dict, pad_idx, args, True, args.actionness, args.use_anchors, args.checkpoint)
        else:
            # 使用通用评估函数
            evaluate(args.split, model, n_class, actions_dict, pad_idx, args, args.overlap, True, args.actionness, args.use_anchors)
    else:
        # 其他数据集分割的评估（返回结果、预测和真实标签）
        if args.dataset == 'soccernetballanticipation':
            results, predictions, targets = evaluate_BAA(args.split, model, n_class, actions_dict, pad_idx, args, True, args.actionness, args.use_anchors, args.checkpoint)
        else:
            results, predictions, targets = evaluate(args.split, model, n_class, actions_dict, pad_idx, args, args.overlap, True, args.actionness, args.use_anchors)
        
        # 打印评估结果
        print(results)
        
        # 保存目标和预测结果用于调试分析
        # 创建结果保存目录（基于检查点文件名）
        os.makedirs(args.checkpoint[:-5]+"-results/", exist_ok=True)
        
        # 保存每个比赛的预测结果
        for game in predictions.keys():
            np.save(args.checkpoint[:-5]+f"-results/predictions-{game.split('/')[-1]}.npy", predictions[game])
        
        # 保存每个比赛的真实标签
        for game in targets.keys():
            np.save(args.checkpoint[:-5]+f"-results/targets-{game.split('/')[-1]}.npy", targets[game])


if __name__ == "__main__":
    main()
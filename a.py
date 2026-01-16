import torch
import torch.nn as nn
import os
import glob
import sys
import argparse
import re
from tqdm import tqdm

# 引入项目中的模块
# 确保此脚本在 FAANTRA 根目录下运行，否则 python 找不到这些模块
from opts import get_args, update_args
from model.futr import FUTR
from eval import evaluate
from eval_BAA import evaluate_BAA
from util.io import load_json
from util.dataset import load_classes

def save_best_model():
    # ---------------------------------------------------------
    # 1. 参数设置与路径解析
    # ---------------------------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="Path to config file (e.g., config/SoccerNetBall/Base-Config-BAA.json)")
    parser.add_argument("ckpt_dir", type=str, help="Directory containing the checkpoint files")
    
    # 临时解析我们需要的参数，忽略其他参数以免冲突
    script_args, _ = parser.parse_known_args()
    
    # 伪造 sys.argv 以便复用 opts.py 中的 get_args()
    sys.argv = [sys.argv[0], script_args.config, "Eval_Best_Search"]
    
    args = get_args()
    config = load_json(args.config)
    args = update_args(args, config)
    
    # 强制使用 GPU
    device = torch.device('cuda')
    args.cpu = False
    args.save_dir = script_args.ckpt_dir # 临时覆盖，方便后续逻辑
    
    print(f"配置文件: {script_args.config}")
    print(f"权重目录: {script_args.ckpt_dir}")

    # ---------------------------------------------------------
    # 2. 初始化模型
    # ---------------------------------------------------------
    # 加载类别
    class_file = os.path.join('data', args.dataset, 'class.txt')
    actions_dict = load_classes(class_file)
    n_class = len(actions_dict) - len(args.excluded_classes)
    pad_idx = 255 

    # 创建 Attention Mask (复制自 main.py)
    src_attn_mask = None
    tgt_attn_mask = None
    if args.mask_attn:
        max_obs_len = int(args.clip_len*max(args.obs_perc))
        src_attn_mask = torch.full((max_obs_len, max_obs_len), True).to(device)
        tgt_attn_mask = torch.full((args.n_query, args.n_query), True).to(device)
        for i in range(max_obs_len):
            start = max(0, i - (args.mask_attn_window_src//2))
            end = min(max_obs_len, i + (args.mask_attn_window_src//2) + 1)
            src_attn_mask[i, start:end] = False
        for i in range(args.n_query):
            start = max(0, i - (args.mask_attn_window_tgt//2))
            end = min(args.n_query, i + (args.mask_attn_window_tgt//2) + 1)
            tgt_attn_mask[i, start:end] = False

    # 实例化模型
    model = FUTR(n_class, args.hidden_dim, device=device, args=args, src_pad_idx=pad_idx,
                            n_query=args.n_query, n_head=args.n_head,
                            num_encoder_layers=args.n_encoder_layer, num_decoder_layers=args.n_decoder_layer,
                            src_attn_mask=src_attn_mask, tgt_attn_mask=tgt_attn_mask).to(device)
    
    model = nn.DataParallel(model).to(device)
    model.eval()

    # ---------------------------------------------------------
    # 3. 寻找并遍历所有 Checkpoint
    # ---------------------------------------------------------
    # 匹配类似 checkpoint14.ckpt, checkpoint31.ckpt 的文件
    # 使用数字排序，而不是默认的字典序 (防止 checkpoint10 排在 checkpoint2 前面)
    ckpt_files = glob.glob(os.path.join(script_args.ckpt_dir, "checkpoint*.ckpt"))
    
    # 提取数字进行排序的辅助函数
    def extract_epoch(filename):
        match = re.search(r'checkpoint(\d+)\.ckpt', filename)
        return int(match.group(1)) if match else -1

    ckpt_files.sort(key=extract_epoch)
    
    if not ckpt_files:
        print("错误: 在指定目录下没有找到任何 'checkpoint*.ckpt' 文件。")
        return

    print(f"找到 {len(ckpt_files)} 个模型权重，准备开始评估...")
    
    best_mAP = -1.0
    best_ckpt_path = ""
    best_epoch_num = -1

    # ---------------------------------------------------------
    # 4. 循环评估
    # ---------------------------------------------------------
    for ckpt_path in tqdm(ckpt_files, desc="Evaluation Progress"):
        current_epoch = extract_epoch(os.path.basename(ckpt_path))
        
        try:
            # 加载权重
            checkpoint = torch.load(ckpt_path, map_location=device)
            
            # 处理 checkpoint 字典结构
            if "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
            else:
                model.load_state_dict(checkpoint)

            # 执行评估 (验证集)
            # 根据 dataset 类型选择评估函数
            if args.dataset == 'soccernetballanticipation':
                eval_res, _, _ = evaluate_BAA("val", model, n_class, actions_dict, pad_idx, args, 
                                              False, use_actionness=args.actionness, use_anchors=args.use_anchors)
            else:
                eval_res, _, _ = evaluate("val", model, n_class, actions_dict, pad_idx, args, 
                                          0.9 if args.n_query == 1 else 1-args.pred_perc, 
                                          False, use_actionness=args.actionness, use_anchors=args.use_anchors)
            
            # 获取核心指标: tightV2 -> a_mAP_stable (原始代码逻辑)
            # 如果没有 tightV2，尝试兼容其他 metric
            if "tightV2" in eval_res and "a_mAP_stable" in eval_res["tightV2"]:
                current_mAP = eval_res["tightV2"]["a_mAP_stable"]
            else:
                # 兼容可能的其他返回格式
                print(f"Warning: 'tightV2' metric not found in results for {os.path.basename(ckpt_path)}. Using default 0.")
                current_mAP = 0.0
            
            tqdm.write(f"Epoch {current_epoch}: mAP = {current_mAP:.5f}")

            # 更新最佳记录
            if current_mAP > best_mAP:
                best_mAP = current_mAP
                best_ckpt_path = ckpt_path
                best_epoch_num = current_epoch
                tqdm.write(f"--> 发现新高! New Best mAP: {best_mAP:.5f} (Epoch {current_epoch})")

        except Exception as e:
            tqdm.write(f"Error loading/evaluating {ckpt_path}: {e}")
            continue

    # ---------------------------------------------------------
    # 5. 保存结果
    # ---------------------------------------------------------
    print("\n" + "="*60)
    if best_ckpt_path:
        print(f"评估完成！")
        print(f"最佳模型来自: {os.path.basename(best_ckpt_path)}")
        print(f"最佳 mAP (stable): {best_mAP:.5f}")
        
        # 按照原始代码逻辑，best.ckpt 只包含 state_dict
        best_checkpoint_content = torch.load(best_ckpt_path, map_location='cpu')
        
        if "model_state_dict" in best_checkpoint_content:
            state_dict_to_save = best_checkpoint_content["model_state_dict"]
        else:
            state_dict_to_save = best_checkpoint_content
            
        save_path = os.path.join(script_args.ckpt_dir, "best_checkpoint.ckpt") # 保持原名 best_checkpoint.ckpt
        torch.save(state_dict_to_save, save_path)
        
        print(f"已保存为: {save_path}")
        print("现在你可以使用这个权重进行测试了。")
    else:
        print("未能找到最佳模型。")
    print("="*60)

if __name__ == '__main__':
    save_best_model()
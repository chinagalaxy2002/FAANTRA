import os
import numpy as np
import torch
import copy
from tqdm import tqdm  # 进度条显示库
from math import ceil  # 向上取整函数
from torch.utils.data import DataLoader  # PyTorch数据加载器
from collections import defaultdict  # 默认字典，用于统计数据
from tabulate import tabulate  # 表格格式化输出库
from dataset.datasets import STRIDE_SNBA  # 导入SoccerNet Ball Anticipation的帧步长常量
from dataset.frame import ActionAnticipationVideoDataset, FPS_SN  # 导入数据集类和帧率常量
from util.io import store_json_snba  # 导入结果存储函数
from SoccerNet.Evaluation.ActionSpotting import average_mAP  # 导入mAP计算函数

# 常量定义
INFERENCE_BATCH_SIZE = 4  # 推理时的批次大小

# 错误统计类 - 用于计算帧级别的分类错误率
class ErrorStat:

    def __init__(self):
        self._total = 0  # 总帧数
        self._err = 0    # 错误帧数

    def update(self, true, pred):
        """更新错误统计"""
        self._err += np.sum(true != pred)  # 累计预测错误的帧数
        self._total += true.shape[0]*true.shape[1]  # 累计总帧数 (clip数 × 每个clip的帧数)

    def get(self):
        """返回错误率"""
        if self._total == 0: return 0
        return self._err / self._total

    def get_acc(self):
        return 1. - self.get()
    
# 前景F1分数计算类 - 用于计算动作识别的F1分数
class ForegroundF1:

    def __init__(self):
        # 使用defaultdict存储每个类别的TP/FP/FN
        self._tp = defaultdict(int)  # True Positive: 正确预测为动作
        self._fp = defaultdict(int)  # False Positive: 错误预测为动作
        self._fn = defaultdict(int)  # False Negative: 漏检动作

    def update(self, true, pred):
        """更新F1统计 - 逐帧逐类别统计"""
        if pred != 0:  # 如果预测为动作（非背景）
            if true != 0:  # 真实也是动作
                self._tp[None] += 1  # None键用于存储"任意动作"的统计
            else:  # 真实是背景
                self._fp[None] += 1  # 假阳性

            if pred == true:  # 预测类别完全正确
                self._tp[pred] += 1  # 该类别的TP加1
            else:  # 预测类别错误
                self._fp[pred] += 1  # 预测类别的FP加1
                if true != 0:  # 如果真实标签也是动作
                    self._fn[true] += 1  # 真实类别的FN加1
        elif true != 0:  # 预测为背景，但真实是动作
            self._fn[None] += 1  # 任意动作的FN加1
            self._fn[true] += 1  # 真实类别的FN加1

    def get(self, k):
        """获取类别k的F1分数"""
        return self._f1(k)

    def tp_fp_fn(self, k):
        """获取类别k的TP/FP/FN统计"""
        return self._tp[k], self._fp[k], self._fn[k]

    def _f1(self, k):
        """计算F1分数公式: TP / (TP + 0.5*FP + 0.5*FN)"""
        denom = self._tp[k] + 0.5 * self._fp[k] + 0.5 * self._fn[k]
        if denom == 0:  # 避免除零错误
            # assert self._tp[k] == 0
            denom = 1
        return self._tp[k] / denom

# 处理帧级别预测结果
def process_frame_predictions(pred_dict, target_labels, pad_idx):
    """
    将原始预测分数转换为最终预测类别，并计算评估指标
    """

    err = ErrorStat()  # 错误率统计
    f1 = ForegroundF1()  # F1分数统计（忽略EOS/背景）

    pred_scores = {}  # 存储归一化后的预测分数
    for video, (scores, support) in (sorted(pred_dict.items())):
        label = target_labels[video]  # 获取该视频的真实标签
        
        # 处理support为0的情况（避免除零）
        if np.min(support) == 0:
            support[support == 0] = 1
        assert np.min(support) > 0, (video, support.tolist())
        
        # 归一化分数
        scores /= support[..., None]
        
        # 找出没有任何预测的帧
        indices_to_pad = scores.sum(axis=-1) == 0
        
        # 获取预测类别
        pred = np.argmax(scores, axis=-1)
        
        # 将没有预测的帧标记为pad_idx
        pred[indices_to_pad] = pad_idx
        
        # 更新错误率和F1统计
        err.update(label, pred)

        pred_scores[video] = scores  # 保存归一化后的分数
        
        # 逐帧更新F1统计
        for i in range(pred.shape[0]):  # 遍历所有clip
            for j in range(pred.shape[1]):  # 遍历每个clip的所有帧
                f1.update(label[i,j], pred[i,j])
    
    return err, f1, pred_scores

# 主评估函数 - 评估球动作预期任务(Ball Action Anticipation)
def evaluate_BAA(split, model, n_class, classes_dict, pad_index, args, test=False, use_actionness=False, use_anchors=False, save_pred = None):
    print(f"\n{'='*20} 开始评估 evaluate_BAA ({split}) {'='*20}")
    model.eval() 
    
    split_path = os.path.join('data', args.dataset, f'{split}.json')
    print(f"[Debug] 数据集路径: {split_path}")
    
    if os.path.exists(split_path):
        EOS_index = 0 
        
        obs_len = int(args.clip_len*args.test_obs_perc)
        pred_len = ceil(5*FPS_SN/STRIDE_SNBA)
        print(f"[Debug] 观测帧数(obs_len): {obs_len}, 预测长度(pred_len): {pred_len}")
        
        pred_dict = {} 
        actual_labels = {} 
        actual_visibility = {} 
        target_labels = {} 
        target_visibility = {} 
        
        jointtrain_exists = args.jointtrain is not None
        model_head1_sizes = [n_class, n_class - 1*args.actionness]
        print(f"[Debug] 联合训练: {jointtrain_exists}, 类别数: {n_class}, Head1大小: {model_head1_sizes}")

        split_data = ActionAnticipationVideoDataset(
            classes_dict, split_path, args.frame_dir, 
            obs_len, stride=STRIDE_SNBA, dataset=args.dataset
        )
        print(f"[Debug] 数据集加载完毕，共包含 {len(split_data)} 个样本 (Clips)")
        
        if split_data._dataset == 'soccernetball':
            raise NotImplementedError(f'To evaluate on the soccernetball dataset use the eval.py file.')
        elif not split_data._dataset == 'soccernetballanticipation':
            raise NotImplementedError(f'Evaluation for {split_data._dataset} is not implemented yet.')
        
        # 初始化存储空间
        print("[Debug] 初始化全局存储字典...")
        for video, _, num_clip, _ in split_data.videos:
            pred_dict[video] = (
                np.zeros((num_clip, pred_len, n_class), np.float32), 
                np.zeros((num_clip, pred_len), np.int32) 
            )
            actual_labels[video] = np.zeros((num_clip, pred_len), np.int32)
            actual_visibility[video] = np.zeros((num_clip, pred_len), np.int32)
            target_labels[video] = np.ones_like(actual_labels[video]) * pad_index
            target_visibility[video] = np.zeros_like(actual_visibility[video])
        print(f"[Debug] 全局字典初始化完成，共 {len(pred_dict)} 个视频")

        with torch.no_grad():
            print("[Debug] 进入推理循环 (DataLoader)...")
            for clip in tqdm(DataLoader(
                    split_data, num_workers=4 * 2, pin_memory=True,
                    batch_size=INFERENCE_BATCH_SIZE, shuffle=False
            )):
                
                outputs = model(clip['frame'][:,:obs_len], mode="test")
                
                if jointtrain_exists:
                    batch_pred_scores = outputs['action'][...,:model_head1_sizes[1]].softmax(dim=2).detach().cpu().numpy()
                else:
                    batch_pred_scores = outputs['action'].softmax(dim=2).detach().cpu().numpy()
                
                # 处理动作性分数（actionness）
                if use_actionness:
                    if jointtrain_exists:
                        batch_actionness = outputs['actionness'][...,:args.n_query].sigmoid().detach().cpu().numpy()
                    else:
                        batch_actionness = outputs['actionness'].sigmoid().detach().cpu().numpy()
                    
                    # [修复] 检查形状，防止重复添加背景类
                    if batch_pred_scores.shape[-1] < n_class:
                        # 原逻辑：需要手动添加背景类（索引0）
                        temp = np.zeros((batch_pred_scores.shape[0], batch_pred_scores.shape[1], 1), batch_pred_scores.dtype)
                        batch_pred_scores = np.concatenate((temp, batch_pred_scores), axis=2)
                        
                        # 应用actionness (乘到所有类上，因为背景是0)
                        if args.loss_func != "BCE":
                            for i in range(batch_pred_scores.shape[2]):
                                batch_pred_scores[...,i] *= batch_actionness
                    else:
                        # 新逻辑：模型输出已经包含背景类 (11类)
                        # 只将 actionness 应用于动作类 (1-10)，不影响背景类 (0)
                        if args.loss_func != "BCE":
                            for i in range(1, batch_pred_scores.shape[2]):
                                batch_pred_scores[...,i] *= batch_actionness

                if jointtrain_exists:
                    batch_pred_offsets = outputs['offset'][...,:args.n_query].detach().cpu().numpy()
                else:
                    batch_pred_offsets = outputs['offset'].detach().cpu().numpy()

                # 转换：Query预测 -> 时间轴预测
                batch_seg_scores = np.zeros((batch_pred_scores.shape[0], pred_len, batch_pred_scores.shape[2]), batch_pred_scores.dtype)
                
                if use_anchors:
                    max_offset = ceil(pred_len / args.n_query)
                    for i in range(batch_pred_scores.shape[0]): 
                        for j in range(batch_pred_scores.shape[1]): 
                            if batch_pred_offsets[i, j] < 0 or batch_pred_offsets[i, j] >= max_offset:
                                continue
                            elif np.argmax(batch_pred_scores[i, j]) == EOS_index:
                                if args.anticipate_background: continue
                            elif batch_seg_scores[i, int(j*max_offset + batch_pred_offsets[i, j])].sum() > 0:
                                batch_seg_scores[i, int(j*max_offset + batch_pred_offsets[i, j])] += batch_pred_scores[i, j]
                                batch_seg_scores[i, int(j*max_offset + batch_pred_offsets[i, j])] /= 2
                            else: 
                                batch_seg_scores[i, int(j*max_offset + batch_pred_offsets[i, j])] = batch_pred_scores[i, j]
                else: # 无 Anchor 模式
                    for i in range(batch_pred_scores.shape[0]): 
                        for j in range(batch_pred_scores.shape[1]): 
                            if batch_pred_offsets[i, j] < 0 or batch_pred_offsets[i, j] >= pred_len:
                                continue
                            elif np.argmax(batch_pred_scores[i, j]) == EOS_index:
                                if args.anticipate_background: continue
                                else: break 
                            elif batch_seg_scores[i, int(batch_pred_offsets[i, j])].sum() > 0:
                                batch_seg_scores[i, int(batch_pred_offsets[i, j])] += batch_pred_scores[i, j]
                                batch_seg_scores[i, int(batch_pred_offsets[i, j])] /= 2
                            else:
                                batch_seg_scores[i, int(batch_pred_offsets[i, j])] = batch_pred_scores[i, j]
                
                # 聚合到全局字典
                for i in range(clip['frame'].shape[0]):
                    video = clip['video'][i] 
                    clip_num = clip['clip_num'][i] 
                    
                    actual_labels[video][clip_num, ...] = clip['label'][i]
                    actual_visibility[video][clip_num, ...] = clip['visibility'][i]
                    
                    scores, support = pred_dict[video]
                    pred_scores = batch_seg_scores[i]

                    # [修复] 此时 pred_scores 形状应该和 scores 一致
                    if pred_scores.shape[-1] != scores.shape[-1]:
                        # 如果仍有不一致，做最后的安全截断或填充
                        if pred_scores.shape[-1] > scores.shape[-1]:
                             pred_scores = pred_scores[..., :scores.shape[-1]]
                        else:
                             # 填充
                             diff = scores.shape[-1] - pred_scores.shape[-1]
                             pred_scores = np.concatenate([pred_scores, np.zeros((*pred_scores.shape[:-1], diff))], axis=-1)

                    scores[clip_num, ...] += pred_scores
                    # 记录该帧被预测了多少次
                    support[clip_num, :] += (pred_scores.sum(axis=1) != 0) * 1
                    
                    for label_index in range(pred_scores.shape[0]):
                        if ((actual_labels[video][clip_num][label_index] != pad_index) and 
                            (actual_labels[video][clip_num][label_index] != 0) and 
                            (actual_labels[video][clip_num][label_index] not in args.excluded_classes)):
                            target_labels[video][clip_num][label_index] = actual_labels[video][clip_num][label_index]
                            target_visibility[video][clip_num][label_index] = actual_visibility[video][clip_num][label_index]
                

        print("[Debug] 推理循环结束，开始处理最终指标...")
        err, f1, pred_scores = process_frame_predictions(pred_dict, target_labels, pad_index)
        print(f"[Debug] process_frame_predictions 完成. 错误率: {err.get()}")
        
        if not test: 
            if split_data._dataset == 'soccernetballanticipation':
                return evaluate_SNBA(target_labels, target_visibility, pred_scores, ((pred_len*2)//(FPS_SN/STRIDE_SNBA))+1)
            else:
                raise NotImplementedError(f'Evaluation for {split_data._dataset} is not implemented yet.')
        else: 
            if split != 'challenge': 
                print('=== Results on {} (w/o NMS) ==='.format(split))
                print('Error (frame-level): {:0.2f}\n'.format(err.get() * 100))

                def get_f1_tab_row(str_k):
                    k = classes_dict[str_k] if str_k != 'any' else None
                    return [str_k, f1.get(k) * 100, *f1.tp_fp_fn(k)]
                
                rows = [get_f1_tab_row('any')] 
                for c in sorted(classes_dict): 
                    rows.append(get_f1_tab_row(c))

                print(tabulate(rows, headers=['Exact frame', 'F1', 'TP', 'FP', 'FN'],
                                floatfmt='0.2f'))
                print()
                
                if save_pred is not None:
                    print(f"Storing test predictions somewhere under {os.path.join('/'.join(save_pred.split('/')[:-1]) + '/preds')}")
                    store_json_snba(save_pred, pred_scores, pad_index, classes_dict, STRIDE_SNBA)
                
                return evaluate_SNBA(target_labels, target_visibility, pred_scores, ((pred_len*2)//(FPS_SN/STRIDE_SNBA))+1)
            else: 
                if save_pred is not None:
                    print(f"Storing challenge predictions somewhere under {os.path.join('/'.join(save_pred.split('/')[:-1]) + '/preds')}")
                    store_json_snba(save_pred, pred_scores, pad_index, classes_dict, STRIDE_SNBA)
                else:
                    print("No path for storing has been given. Will not store challenge predictions")



# SoccerNet Ball Anticipation评估的包装函数
def evaluate_SNBA(target_labels, target_visibility, pred_scores, max_mAP):
    """调用多指标评估函数"""
    return multi_aux_evaluate(target_labels, target_visibility, pred_scores, max_mAP, version=2, framerate=FPS_SN/STRIDE_SNBA)

# 评估多个mAP指标
def multi_aux_evaluate(target_labels, target_visibility, pred_scores, max_mAP, version=2, framerate=FPS_SN):
    """
    计算多个不同容差(delta)下的mAP
    """
    res = {}
    for metric in ["at1", "at2", "at3", "at4", "at5", ["atInfty", [max_mAP]], "tight", ["tightV2", [1, 5, max_mAP]]]:
        if isinstance(metric, str):  # 预定义的指标名称
            res[metric], targets = aux_evaluate(target_labels, target_visibility, pred_scores, version, framerate, metric)
        else:  # 自定义容差列表
            res[metric[0]], targets = aux_evaluate(target_labels, target_visibility, pred_scores, version, framerate, metric[1])
    
    # 为向后兼容性添加别名
    for k, v in res.items():
        v["a_mAP_stable"] = v["a_mAP"]
    
    return res, pred_scores, targets

# 辅助评估函数 - 核心mAP计算
def aux_evaluate(target_labels, target_visibility, pred_scores, version=2, framerate=FPS_SN, metric="loose"):
    """
    将数据转换为SoccerNet评估脚本所需格式，并计算mAP
    """

    # 初始化列表（每个元素是一个clip的数据）
    targets_numpy = list()  # 真实标签矩阵
    detections_numpy = list()  # 预测分数矩阵
    closests_numpy = list()  # 最近动作矩阵（用于某些mAP变体）
        
    targets_save = {}  # 用于保存目标（调试用）

    # 处理每个视频
    for video in tqdm(list(pred_scores.keys())):

        # 减去1e-6：修正SoccerNet评估脚本中的阈值处理问题
        predictions_game = pred_scores[video] - 1e-6
        
        # 创建目标矩阵：与预测矩阵同shape，值为可见性
        targets_game = np.zeros_like(predictions_game)
        for c in range(targets_game.shape[0]):  # 遍历每个clip
            for i in range(targets_game.shape[1]):  # 遍历每帧
                # 确保标签索引有效
                if target_labels[video][c][i] < targets_game.shape[1]:
                    # 在对应类别位置填入可见性值
                    targets_game[c, i, target_labels[video][c][i]] = target_visibility[video][c][i]

        targets_save[video] = targets_game

        # 创建"最近动作"矩阵
        closest_numpy = np.zeros(targets_game.shape) - 1
        
        for clip in np.arange(targets_game.shape[0]):
            for c in np.arange(targets_game.shape[-1]):
                indexes = np.where(targets_game[clip][:, c] != 0)[0].tolist()
                if len(indexes) == 0:  
                    continue
                
                indexes.insert(0, -indexes[0])  # 左边界
                indexes.append(2 * closest_numpy.shape[0])  # 右边界
                
                for i in np.arange(len(indexes) - 2) + 1:
                    start = max(0, (indexes[i - 1] + indexes[i]) // 2)
                    stop = min(closest_numpy.shape[0], (indexes[i] + indexes[i + 1]) // 2)
                    closest_numpy[clip][start:stop, c] = targets_game[clip][indexes[i], c]

        for clip in range(targets_game.shape[0]):
            # 注意：移除背景类（索引0）
            if targets_game.shape[-1] > 1:
                targets_numpy.append(targets_game[clip,:,1:])
                detections_numpy.append(predictions_game[clip,:,1:])
                closests_numpy.append(closest_numpy[clip,:,1:])
            else:
                 # Fallback if somehow only 1 class
                 targets_numpy.append(targets_game[clip,:,:])
                 detections_numpy.append(predictions_game[clip,:,:])
                 closests_numpy.append(closest_numpy[clip,:,:])


    if metric == "loose":  
        deltas = np.arange(12) * 5 + 5
    elif metric == "tight":  
        deltas = np.arange(5) * 1 + 1
    elif metric == "at1":  
        deltas = np.array([1])
    elif metric == "at2":  
        deltas = np.array([2])
    elif metric == "at3":  
        deltas = np.array([3])
    elif metric == "at4":  
        deltas = np.array([4])
    elif metric == "at5":  
        deltas = np.array([5])
    elif isinstance(metric, list):  
        deltas = np.array(metric)
    
    a_mAP, a_mAP_per_class, a_mAP_visible, a_mAP_per_class_visible, a_mAP_unshown, a_mAP_per_class_unshown = (
        average_mAP(targets_numpy, detections_numpy, closests_numpy, framerate, deltas=deltas)
    )

    results = {
        "a_mAP": a_mAP,  
        "a_mAP_per_class": a_mAP_per_class,  
        "a_mAP_visible": a_mAP_visible if version == 2 else None,  
        "a_mAP_per_class_visible": a_mAP_per_class_visible if version == 2 else None,
        "a_mAP_unshown": a_mAP_unshown if version == 2 else None,  
        "a_mAP_per_class_unshown": a_mAP_per_class_unshown if version == 2 else None,
    }
    return results, targets_save
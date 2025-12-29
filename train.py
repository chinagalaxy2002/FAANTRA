import torch #(v3)
import torch.nn.functional as F
import os
import wandb
from dataset.datasets import STRIDE_SNB
from dataset.frame import FPS_SN
from tqdm import tqdm
from utils import cal_performance, normalize_offset, cal_actionness_performance, CALF_matching, CALF_matching2
from eval import evaluate
from eval_BAA import evaluate_BAA

# Segmentation loss is forced to can only use CE as loss function
def train(args, model, train_loader, val_loader, optimizer, scheduler, criterion,
          model_save_path, pad_idx, device, num_pred_frames, n_class, class_dict, n_query,
          start_epoch=0, offset_loss_weight=1.0, use_actionness=False, use_anchors=False,
          loss_func="CE", best_mAP = 0, best_model_path=""):
    torch.autograd.set_detect_anomaly(True)
    inv_class_dict = {v: k for k, v in class_dict.items()}
    num_pred_frames = int(num_pred_frames // n_query) if use_anchors else num_pred_frames
    BCE_with_actionness = loss_func == "BCE" and use_actionness
    if BCE_with_actionness: use_actionness = False
    model.to(device)
    model.train()
    best_mAP = best_mAP
    best_model_path = best_model_path
    print("Training Start")
    for epoch in range(start_epoch, args.epochs):

        ########################################
        # Training
        ########################################

        epoch_loss = 0
        epoch_loss_class = 0
        epoch_loss_off = 0
        epoch_loss_seg = 0
        epoch_class_stats = None
        epoch_class_stats_seg = None
        if use_actionness:
            epoch_loss_actionness = 0
            epoch_actionness_stats = None
        total_class = 0
        total_class_correct = 0
        total_off_correct = 0
        total_seg = 0
        total_seg_correct = 0
        
        train_loop = tqdm(train_loader)
        
        for i, data in enumerate(train_loop):
            step_log_dict = {"train/step": epoch*len(train_loader) + i+1}
            postfix_kwargs = {"loss": 0}
            optimizer.zero_grad()
            
            # 1. 解包
            features, past_label, trans_off_future, trans_future_target, target_actionness = data
            features = features.to(device)
            past_label = past_label.to(device)
            trans_off_future = trans_off_future.to(device)
            trans_future_target = trans_future_target.to(device)
            target_actionness = target_actionness.to(device)

            # 2. 预处理 Targets
            trans_off_future_mask = (trans_off_future != pad_idx).long().to(device) 
            target_off = trans_off_future * trans_off_future_mask 
            target = trans_future_target
            
            # 3. 构造 Diffusion 输入 (适配 SeqConcat 架构)
            # -----------------------------------------------------------------
            # 创建 Future Padding Mask (B, T_future, 1)
            future_mask = (target != pad_idx).unsqueeze(-1).float()

            # 构造 x_0 (Target converted to One-Hot)
            safe_target = target.clone()
            safe_target[target == pad_idx] = 0 
            
            x_0_onehot = F.one_hot(safe_target.long(), num_classes=n_class).float() 
            
            # [修改] 构造 masks_stages
            # 不再重复 channel，保持 (B, T, 1)，利用广播机制匹配 Diffusion 输出的 13 通道
            masks_stages = [future_mask]

            # 组装 Batch 字典
            inputs = {
                'x_0': x_0_onehot,
                'obs': features,            # Past Features
                'mask_past': future_mask,   # Future Padding Mask
                'masks_stages': masks_stages,
                
                # 辅助任务数据
                'action_target': target, 
                'offset_target': target_off,
                'actionness_target': target_actionness if (use_actionness or BCE_with_actionness) else None,
                'past_label': past_label,
                'mode': 'train'
            }
            # -----------------------------------------------------------------

            # 4. Forward
            outputs = model(inputs)
            losses = 0

            # Diffusion 逻辑判断
            is_diffusion = 'loss' in outputs
            
            if is_diffusion:
                diff_loss = outputs['loss']
                losses += diff_loss
                step_log_dict["train/diff_loss"] = diff_loss.item()
                postfix_kwargs["loss_diff"] = diff_loss.item()
            
            # Seg Task
            if args.seg and 'seg' in outputs:
                output_seg = outputs['seg']
                B, T, C = output_seg.size()
                output_seg = output_seg.reshape(-1, C).to(device)
                target_past_label = past_label.reshape(-1) 
                class_weights = torch.tensor(args.class_weights, device=device)
                loss_seg, n_seg_correct, n_seg_total, seg_class_stats = cal_performance(output_seg, target_past_label, pad_idx, class_weights=class_weights)
                losses += loss_seg
                total_seg += n_seg_total
                total_seg_correct += n_seg_correct
                epoch_loss_seg += loss_seg.item()
                step_log_dict["train/seg_loss"] = loss_seg.item()
                
                if n_seg_total == 0:
                    step_log_dict["train/seg_acc"] = 1
                else:
                    step_log_dict["train/seg_acc"] = n_seg_correct/n_seg_total
                    
                if epoch_class_stats_seg is None:
                    epoch_class_stats_seg = seg_class_stats
                else:
                    for class_key in seg_class_stats.keys():
                        for class_stat in seg_class_stats[class_key].keys():
                            epoch_class_stats_seg[class_key][class_stat] += seg_class_stats[class_key][class_stat]
            
            # Anticipation Task
            if args.anticipate:
                do_calc_loss = not is_diffusion 

                output = outputs['action'] 
                
                if args.CALF_matching:
                    if args.CALF_probability_matching:
                        output, output_off, output_actionness = CALF_matching2(output, target, outputs.get('offset'), target_off, pad_idx, use_actionness=use_actionness,
                                                                               output_actionness=outputs.get('actionness') if use_actionness else None,
                                                                               target_actionness=target_actionness if use_actionness else None)
                    else:
                        output, output_off, output_actionness = CALF_matching(output, target, outputs.get('offset'), target_off, pad_idx, use_actionness=use_actionness,
                                                                              output_actionness=outputs.get('actionness') if use_actionness else None,
                                                                              target_actionness=target_actionness if use_actionness else None)
                else:
                    output_off = outputs.get('offset')
                    if use_actionness: output_actionness = outputs.get('actionness')

                B, T, C = output.size()
                output = output.reshape(-1, C).to(device)
                target = target.contiguous().view(-1)

                target = torch.where(target == pad_idx, target, target - 1) if use_actionness or BCE_with_actionness else target
                class_weights = torch.tensor(args.class_weights, device=device)
                class_weights[0] = args.eos_weight
                
                if use_actionness or BCE_with_actionness:
                    output_for_loss = output[:, 1:]
                    loss, n_correct, n_total, class_stats = cal_performance(output_for_loss, target, pad_idx, loss_func=loss_func, class_weights=class_weights[1:], actionness=True, calc_loss=do_calc_loss)
                    
                    new_stats = {}
                    for k, v in class_stats.items():
                        new_stats[k+1] = v
                    class_stats = new_stats

                else:
                    loss, n_correct, n_total, class_stats = cal_performance(output, target, pad_idx, loss_func=loss_func, class_weights=class_weights, actionness=use_actionness, calc_loss=do_calc_loss)
                
                acc = 1 if n_total == 0 else n_correct / n_total
                
                if not is_diffusion and loss is not None:
                    loss = torch.nan_to_num(loss)
                    losses += loss
                    step_log_dict["train/CE_loss"] = loss.item()
                    postfix_kwargs["loss_CE"] = loss.item()
                    epoch_loss_class += loss.item()

                total_class += n_total
                total_class_correct += n_correct
                
                # Offset processing
                if output_off is not None:
                    output_off = output_off if args.CALF_matching else outputs['offset']
                    
                    if not is_diffusion:
                        transformed_output_off = normalize_offset(output_off, trans_off_future_mask, num_pred_frames)
                        transformed_target_off = normalize_offset(target_off, trans_off_future_mask, num_pred_frames)
                        
                        if torch.sum(trans_off_future_mask) == 0:
                            loss_off = torch.sum(criterion(transformed_output_off, transformed_target_off))
                        else:
                            loss_off = torch.sum(criterion(transformed_output_off, transformed_target_off)) / torch.sum(trans_off_future_mask)
                        loss_off *= offset_loss_weight
                        losses += loss_off
                        epoch_loss_off += loss_off.item()
                        step_log_dict["train/offset_loss"] = loss_off.item()

                    # Offset Accuracy
                    unrolled_output_off = output_off.reshape(-1)
                    unrolled_target_off = target_off.view(-1)
                    unrolled_off_mask = trans_off_future_mask.view(-1)
                    off_correct = 0
                    for d in range(len(unrolled_target_off)):
                        if unrolled_off_mask[d]:
                            if (unrolled_output_off[d] - unrolled_target_off[d]).abs() <= FPS_SN/STRIDE_SNB:
                                off_correct += 1
                    total_off_correct += off_correct
                    step_log_dict["train/offset_acc@1s"] = 1 if n_total == 0 else off_correct/n_total
                    postfix_kwargs["acc_offset"] = 1 if n_total == 0 else off_correct/n_total
                
                # Actionness
                if use_actionness and output_actionness is not None:
                    output_actionness = output_actionness if args.CALF_matching else outputs['actionness']
                    output_actionness = output_actionness.reshape(-1).to(device)
                    target_actionness = target_actionness.contiguous().view(-1)
                    
                    actionness_loss, actionness_stats = cal_actionness_performance(output_actionness, target_actionness, threshold=0.5)
                    
                    if not is_diffusion:
                        losses += actionness_loss
                        epoch_loss_actionness += actionness_loss.item()
                    
                    total = actionness_stats["TP"] + actionness_stats["TN"] + actionness_stats["FP"] + actionness_stats["FN"]
                    step_log_dict["train/actionness_acc"] = (actionness_stats["TP"] + actionness_stats["TN"]) / total if total > 0 else 0
                    postfix_kwargs["acc_actionness"] = (actionness_stats["TP"] + actionness_stats["TN"]) / (total + 1e-8)
                    
                    if epoch_actionness_stats is None:
                        epoch_actionness_stats = actionness_stats
                    else:
                        for stat_key in actionness_stats.keys():
                            epoch_actionness_stats[stat_key] += actionness_stats[stat_key]

                # Stats update
                step_log_dict["train/CE_acc"] = acc
                postfix_kwargs["acc_ant"] = acc

                if epoch_class_stats is None:
                    epoch_class_stats = class_stats
                else:
                    for class_key in class_stats.keys():
                        for class_stat in class_stats[class_key].keys():
                            epoch_class_stats[class_key][class_stat] += class_stats[class_key][class_stat]
                

            # Backprop
            epoch_loss += losses.item()
            losses.backward()
            optimizer.step()
            
            step_log_dict["train/full_loss"] = losses.item()
            postfix_kwargs["loss"] = losses.item()
            train_loop.set_description(f"Epoch [{epoch+1}/{args.epochs}]")
            train_loop.set_postfix(**postfix_kwargs)
            step_log_dict["train/lr"] = optimizer.param_groups[0]['lr']
            
            if args.seg:
                inv_class_dict[0] = "BACKGROUND"
                step_log_dict = log_class_metrics(step_log_dict, seg_class_stats, "train/seg", inv_class_dict)
            if args.anticipate:
                inv_class_dict[0] = "EOS"
                step_log_dict = log_class_metrics(step_log_dict, class_stats, "train/anticipate", inv_class_dict)
            wandb.log(step_log_dict)
            scheduler.step()


        ########################################
        # Validation
        ########################################

        val_epoch_loss = 0
        val_epoch_loss_class = 0
        val_epoch_loss_off = 0
        val_epoch_loss_seg = 0
        val_epoch_class_stats = None
        val_epoch_class_stats_seg = None
        if use_actionness:
            val_epoch_loss_actionness = 0
            val_epoch_actionness_stats = None
        val_total_class = 0
        val_total_class_correct = 0
        val_total_off_correct = 0
        val_total_seg = 0
        val_total_seg_correct = 0
        val_loop = tqdm(val_loader)
        
        with torch.no_grad():
            for j, data in enumerate(val_loop):
                step_log_dict = {"val/step": epoch*len(val_loader) + j+1}
                postfix_kwargs = {"loss": 0}
                features, past_label, trans_off_future, trans_future_target, target_actionness = data
                features = features.to(device)
                past_label = past_label.to(device)
                trans_off_future = trans_off_future.to(device)
                trans_future_target = trans_future_target.to(device)
                trans_off_future_mask = (trans_off_future != pad_idx).long().to(device) 
                target_actionness = target_actionness.to(device)

                target_off = trans_off_future*trans_off_future_mask 
                target = trans_future_target

                # 构造 Validation Inputs (与 Train 保持一致)
                # -----------------------------------------------------------
                future_mask = (target != pad_idx).unsqueeze(-1).float()
                safe_target = target.clone()
                safe_target[target == pad_idx] = 0 
                x_0_onehot = F.one_hot(safe_target.long(), num_classes=n_class).float()
                
                # [修改] Validation 中也去掉 channel repeat
                masks_stages = [future_mask]

                inputs = {
                    'x_0': x_0_onehot,
                    'obs': features,
                    'mask_past': future_mask,
                    'masks_stages': masks_stages,
                    'action_target': target,
                    'offset_target': target_off,
                    'actionness_target': target_actionness,
                    'past_label': past_label,
                    'mode': 'validation'
                }
                # -----------------------------------------------------------

                outputs = model(inputs)
                losses = 0
                
                # Diffusion Check for Validation
                is_diffusion = 'loss' in outputs 
                
                # Seg Task
                if args.seg and 'seg' in outputs:
                    output_seg = outputs['seg']
                    B, T, C = output_seg.size()
                    output_seg = output_seg.reshape(-1, C).to(device)
                    target_past_label = past_label.reshape(-1)
                    class_weights = torch.tensor(args.class_weights, device=device)
                    loss_seg, n_seg_correct, n_seg_total, seg_class_stats = cal_performance(output_seg, target_past_label, pad_idx, class_weights=class_weights)
                    losses += loss_seg
                    val_total_seg += n_seg_total
                    val_total_seg_correct += n_seg_correct
                    val_epoch_loss_seg += loss_seg.item()
                    step_log_dict["val/seg_loss"] = loss_seg.item()
                    
                    if n_seg_total == 0:
                        step_log_dict["val/seg_acc"] = 1
                    else:
                        step_log_dict["val/seg_acc"] = n_seg_correct/n_seg_total
                    
                    if val_epoch_class_stats_seg is None:
                        val_epoch_class_stats_seg = seg_class_stats
                    else:
                        for class_key in seg_class_stats.keys():
                            for class_stat in seg_class_stats[class_key].keys():
                                val_epoch_class_stats_seg[class_key][class_stat] += seg_class_stats[class_key][class_stat]
                
                # Anticipation Task
                if args.anticipate:
                    output = outputs['action']
                    if args.CALF_matching:
                        if args.CALF_probability_matching:
                            output, output_off, output_actionness = CALF_matching2(output, target, outputs.get('offset'), target_off, pad_idx, use_actionness=use_actionness,
                                                                                   output_actionness=outputs.get('actionness') if use_actionness else None,
                                                                                   target_actionness=target_actionness if use_actionness else None)
                        else:
                            output, output_off, output_actionness = CALF_matching(output, target, outputs.get('offset'), target_off, pad_idx, use_actionness=use_actionness,
                                                                                  output_actionness=outputs.get('actionness') if use_actionness else None,
                                                                                  target_actionness=target_actionness if use_actionness else None)
                    else:
                        output_off = outputs.get('offset')
                        if use_actionness: output_actionness = outputs.get('actionness')

                    B, T, C = output.size()
                    
                    output = output.reshape(-1, C).to(device)
                    target = target.contiguous().view(-1)
                    
                    target = torch.where(target == pad_idx, target, target - 1) if use_actionness or BCE_with_actionness else target
                    class_weights = torch.tensor(args.class_weights, device=device)
                    class_weights[0] = args.eos_weight
                    
                    if use_actionness or BCE_with_actionness:
                        output_for_loss = output[:, 1:]
                        loss, n_correct, n_total, class_stats = cal_performance(output_for_loss, target, pad_idx, loss_func=loss_func, class_weights=class_weights[1:], actionness=True)
                        new_stats = {}
                        for k, v in class_stats.items():
                            new_stats[k+1] = v
                        class_stats = new_stats
                    else:
                        loss, n_correct, n_total, class_stats = cal_performance(output, target, pad_idx, loss_func=loss_func, class_weights=class_weights, actionness=use_actionness)
                    
                    acc = 1 if n_total == 0 else n_correct / n_total
                    
                    if loss is not None:
                        loss = torch.nan_to_num(loss)
                        losses += loss
                        val_epoch_loss_class += loss.item()
                        step_log_dict["val/CE_loss"] = loss.item()

                    val_total_class += n_total
                    val_total_class_correct += n_correct

                    # Offset
                    if output_off is not None:
                        output_off = output_off if args.CALF_matching else outputs['offset']
                        transformed_output_off = normalize_offset(output_off, trans_off_future_mask, num_pred_frames)
                        transformed_target_off = normalize_offset(target_off, trans_off_future_mask, num_pred_frames)
                        
                        if torch.sum(trans_off_future_mask) == 0:
                            loss_off = torch.sum(criterion(transformed_output_off, transformed_target_off))
                        else:
                            loss_off = torch.sum(criterion(transformed_output_off, transformed_target_off)) / torch.sum(trans_off_future_mask)
                        loss_off *= offset_loss_weight
                        losses += loss_off
                        val_epoch_loss_off += loss_off.item()
                        
                        unrolled_output_off = output_off.reshape(-1)
                        unrolled_target_off = target_off.view(-1)
                        unrolled_off_mask = trans_off_future_mask.view(-1)
                        off_correct = 0
                        for d in range(len(unrolled_target_off)):
                            if unrolled_off_mask[d]:
                                if (unrolled_output_off[d] - unrolled_target_off[d]).abs() <= 25/STRIDE_SNB:
                                    off_correct += 1
                        val_total_off_correct += off_correct
                        step_log_dict["val/offset_loss"] = loss_off.item()
                        step_log_dict["val/offset_acc@1s"] = 1 if n_total == 0 else off_correct/n_total
                    
                    # Actionness
                    if use_actionness and output_actionness is not None:
                        output_actionness = output_actionness if args.CALF_matching else outputs['actionness']
                        output_actionness = output_actionness.reshape(-1).to(device)
                        target_actionness = target_actionness.contiguous().view(-1)
                        actionness_loss, actionness_stats = cal_actionness_performance(output_actionness, target_actionness, threshold=0.5)
                        losses += actionness_loss
                        val_epoch_loss_actionness += actionness_loss.item()
                        
                        if val_epoch_actionness_stats is None:
                            val_epoch_actionness_stats = actionness_stats
                        else:
                            for stat_key in actionness_stats.keys():
                                val_epoch_actionness_stats[stat_key] += actionness_stats[stat_key]

                    step_log_dict["val/CE_acc"] = acc
                    
                    if val_epoch_class_stats is None:
                        val_epoch_class_stats = class_stats
                    else:
                        for class_key in class_stats.keys():
                            for class_stat in class_stats[class_key].keys():
                                val_epoch_class_stats[class_key][class_stat] += class_stats[class_key][class_stat]


                val_epoch_loss += losses.item()
                step_log_dict["val/full_loss"] = losses.item()
                postfix_kwargs["loss"] = losses.item()
                val_loop.set_description(f"Epoch [{epoch+1}/{args.epochs}]")
                val_loop.set_postfix(**postfix_kwargs)
                if args.seg:
                    inv_class_dict[0] = "BACKGROUND"
                    step_log_dict = log_class_metrics(step_log_dict, seg_class_stats, "val/seg", inv_class_dict)
                if args.anticipate:
                    inv_class_dict[0] = "EOS"
                    step_log_dict = log_class_metrics(step_log_dict, class_stats, "val/anticipate", inv_class_dict)
                wandb.log(step_log_dict)

        ########################################
        # Full epoch statistics
        ########################################
        epoch_loss = epoch_loss / (i+1)
        val_epoch_loss = val_epoch_loss / (j+1)
        print("Epoch [", (epoch+1), '/', args.epochs, '] Loss : %.3f'%epoch_loss, 'Val Loss: %.3f'%val_epoch_loss)
        epoch_log_dict = {"epoch": epoch+1, "epoch/full_loss": epoch_loss, "epoch/val_full_loss": val_epoch_loss}
        if args.anticipate :
            inv_class_dict[0] = "EOS"
            accuracy = total_class_correct/total_class if total_class > 0 else 0
            epoch_loss_class = epoch_loss_class / (i+1)
            val_accuracy = val_total_class_correct/val_total_class if val_total_class > 0 else 0
            val_epoch_loss_class = val_epoch_loss_class / (j+1)
            print('Training Acc :%.3f'%accuracy, 'CE loss :%.3f'%epoch_loss_class )
            print('Val Acc :%.3f'%val_accuracy, 'CE loss :%.3f'%val_epoch_loss_class )
            epoch_log_dict["epoch/CE_acc"] = accuracy
            epoch_log_dict["epoch/CE_loss"] = epoch_loss_class
            epoch_log_dict["epoch/val_CE_acc"] = val_accuracy
            epoch_log_dict["epoch/val_CE_loss"] = val_epoch_loss_class
            epoch_loss_off = epoch_loss_off / (i+1)
            offset_acc = total_off_correct / total_class if total_class > 0 else 0
            val_epoch_loss_off = val_epoch_loss_off / (j+1)
            val_offset_acc = val_total_off_correct / val_total_class if val_total_class > 0 else 0
            print('offset acc:%.5f'%offset_acc,'offset loss: %.5f'%epoch_loss_off)
            print('val offset acc:%.5f'%val_offset_acc,'val offset loss: %.5f'%val_epoch_loss_off)
            epoch_log_dict["epoch/offset_loss"] = epoch_loss_off
            epoch_log_dict["epoch/offset_acc"] = offset_acc
            epoch_log_dict["epoch/val_offset_loss"] = val_epoch_loss_off
            epoch_log_dict["epoch/val_offset_acc"] = val_offset_acc
            epoch_log_dict = log_class_metrics(epoch_log_dict, epoch_class_stats, "epoch/train_anticipate", inv_class_dict)
            epoch_log_dict = log_class_metrics(epoch_log_dict, val_epoch_class_stats, "epoch/val_anticipate", inv_class_dict)
            if use_actionness:
                epoch_loss_actionness = epoch_loss_actionness / (i+1)
                val_epoch_loss_actionness = val_epoch_loss_actionness / (j+1)
                epoch_log_dict["epoch/actionness_loss"] = epoch_loss_actionness
                epoch_log_dict["epoch/val_actionness_loss"] = val_epoch_loss_actionness
                epoch_log_dict = log_confusion_matrix(epoch_log_dict, epoch_actionness_stats, "epoch/train_actionness")
                epoch_log_dict = log_confusion_matrix(epoch_log_dict, val_epoch_actionness_stats, "epoch/val_actionness")

        if args.seg :
            inv_class_dict[0] = "BACKGROUND"
            acc_seg = total_seg_correct / total_seg if total_seg > 0 else 0
            val_acc_seg = val_total_seg_correct / val_total_seg if val_total_seg > 0 else 0
            epoch_loss_seg = epoch_loss_seg / (i+1)
            val_epoch_loss_seg = val_epoch_loss_seg / (j+1)
            print('seg loss :%.3f'%epoch_loss_seg, ', seg acc : %.5f'%acc_seg)
            print('val_seg loss :%.3f'%val_epoch_loss_seg, ', val_seg acc : %.5f'%val_acc_seg)
            epoch_log_dict["epoch/seg_loss"] = epoch_loss_seg
            epoch_log_dict["epoch/seg_acc"] = acc_seg
            epoch_log_dict["epoch/val_seg_loss"] = val_epoch_loss_seg
            epoch_log_dict["epoch/val_seg_acc"] = val_acc_seg
            epoch_log_dict = log_class_metrics(epoch_log_dict, epoch_class_stats_seg, "epoch/train_seg", inv_class_dict)
            epoch_log_dict = log_class_metrics(epoch_log_dict, val_epoch_class_stats_seg, "epoch/val_seg", inv_class_dict)

        epoch_log_dict["epoch/lr"] = optimizer.param_groups[0]['lr']

        ########################################
        # Evaluation and checkpoint saving
        ########################################
        save_path = os.path.join(model_save_path)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        if epoch >= args.start_map_epoch:
            if args.anticipate:
                if args.dataset == 'soccernetballanticipation':
                    maps, _, _ = evaluate_BAA("val", model, n_class, class_dict, pad_idx, args, False, use_actionness or BCE_with_actionness, use_anchors)
                else:
                    maps, _, _ = evaluate("val", model, n_class, class_dict, pad_idx, args, 0.9 if args.n_query == 1 else 1-args.pred_perc, False, use_actionness or BCE_with_actionness, use_anchors)
                for key in maps.keys():
                    print(key)
                    print(maps[key])
                    epoch_log_dict[f"epoch/map_{key}"] = maps[key]
            
            torch.save(model.state_dict(), os.path.join(save_path, f'checkpoint{epoch+1}.ckpt'))
            if maps["tightV2"]["a_mAP_stable"] >= best_mAP:
                print(f"\nSaving new best model at epoch {epoch+1} with mAP {maps['tightV2']['a_mAP_stable']} (+{maps['tightV2']['a_mAP_stable']-best_mAP})\n")
                best_model_path = os.path.join(save_path, 'best_checkpoint.ckpt')
                torch.save(model.state_dict(), best_model_path)
                wandb.save(best_model_path)
                best_mAP = maps["tightV2"]["a_mAP_stable"]
                
        checkpoint_dir = os.path.join(save_path, "checkpoint")
        os.makedirs(checkpoint_dir, exist_ok=True)
        torch.save({
            "epoch": epoch+1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "arguments": args,
            "wandb_run_id": wandb.run.id,
            "best_mAP": best_mAP,
            "best_model_path": best_model_path
        }, os.path.join(checkpoint_dir, "checkpoint.ckpt"))
        wandb.log(epoch_log_dict)
        
    return model, best_model_path

def log_class_metrics(log_dict, class_stats, log_prefix, inv_class_dict):
    for class_key in class_stats.keys():
        if class_key not in inv_class_dict:
            inv_class_dict[class_key] = f"Unknown_Class_{class_key}"
            
        for class_stat in class_stats[class_key].keys():
            log_dict[f"{log_prefix}/{inv_class_dict[class_key]}/{class_stat}"] = class_stats[class_key][class_stat]
        if (class_stats[class_key]["TP"] + class_stats[class_key]["FP"] + class_stats[class_key]["TN"] + class_stats[class_key]["FN"]) == 0:
            continue
        log_dict[f"{log_prefix}/{inv_class_dict[class_key]}/acc"] = (class_stats[class_key]["TP"] + class_stats[class_key]["TN"]) / (class_stats[class_key]["TP"] + class_stats[class_key]["FP"] + class_stats[class_key]["TN"] + class_stats[class_key]["FN"])
        log_dict[f"{log_prefix}/{inv_class_dict[class_key]}/Prevalence"] = (class_stats[class_key]["TP"]) / (class_stats[class_key]["TP"] + class_stats[class_key]["FP"] + class_stats[class_key]["TN"] + class_stats[class_key]["FN"])
        do_TPR = (class_stats[class_key]["TP"] + class_stats[class_key]["FN"]) > 0
        do_TNR = (class_stats[class_key]["TN"] + class_stats[class_key]["FP"]) > 0
        if do_TPR: log_dict[f"{log_prefix}/{inv_class_dict[class_key]}/TPR"] = class_stats[class_key]["TP"] / (class_stats[class_key]["TP"] + class_stats[class_key]["FN"])
        if do_TNR: log_dict[f"{log_prefix}/{inv_class_dict[class_key]}/TNR"] = class_stats[class_key]["TN"] / (class_stats[class_key]["TN"] + class_stats[class_key]["FP"])
        if do_TPR and do_TNR:
            log_dict[f"{log_prefix}/{inv_class_dict[class_key]}/Balanced_Acc"] = (log_dict[f"{log_prefix}/{inv_class_dict[class_key]}/TPR"] + log_dict[f"{log_prefix}/{inv_class_dict[class_key]}/TNR"]) / 2
        do_precision = (class_stats[class_key]["TP"] + class_stats[class_key]["FP"]) > 0
        do_recall = (class_stats[class_key]["TP"] + class_stats[class_key]["FN"]) > 0
        if do_precision: log_dict[f"{log_prefix}/{inv_class_dict[class_key]}/precision"] = (class_stats[class_key]["TP"]) / (class_stats[class_key]["TP"] + class_stats[class_key]["FP"])
        if do_recall: log_dict[f"{log_prefix}/{inv_class_dict[class_key]}/recall"] = (class_stats[class_key]["TP"]) / (class_stats[class_key]["TP"] + class_stats[class_key]["FN"])
        if do_precision and do_recall:
            log_dict[f"{log_prefix}/{inv_class_dict[class_key]}/f1"] = 0 if class_stats[class_key]["TP"] == 0 else 2 * (log_dict[f"{log_prefix}/{inv_class_dict[class_key]}/precision"] * log_dict[f"{log_prefix}/{inv_class_dict[class_key]}/recall"]) / (log_dict[f"{log_prefix}/{inv_class_dict[class_key]}/precision"] + log_dict[f"{log_prefix}/{inv_class_dict[class_key]}/recall"])
    return log_dict

def log_confusion_matrix(log_dict, confusion_matrix, log_prefix):
    for metric in confusion_matrix.keys():
        log_dict[f"{log_prefix}/{metric}"] = confusion_matrix[metric]
    if (confusion_matrix["TP"] + confusion_matrix["FP"] + confusion_matrix["TN"] + confusion_matrix["FN"]) == 0:
        return log_dict
    log_dict[f"{log_prefix}/acc"] = (confusion_matrix["TP"] + confusion_matrix["TN"]) / (confusion_matrix["TP"] + confusion_matrix["FP"] + confusion_matrix["TN"] + confusion_matrix["FN"])
    log_dict[f"{log_prefix}/Prevalence"] = (confusion_matrix["TP"]) / (confusion_matrix["TP"] + confusion_matrix["FP"] + confusion_matrix["TN"] + confusion_matrix["FN"])
    do_TPR = (confusion_matrix["TP"] + confusion_matrix["FN"]) > 0
    do_TNR = (confusion_matrix["TN"] + confusion_matrix["FP"]) > 0
    if do_TPR: log_dict[f"{log_prefix}/TPR"] = confusion_matrix["TP"] / (confusion_matrix["TP"] + confusion_matrix["FN"])
    if do_TNR: log_dict[f"{log_prefix}/TNR"] = confusion_matrix["TN"] / (confusion_matrix["TN"] + confusion_matrix["FP"])
    if do_TPR and do_TNR:
        log_dict[f"{log_prefix}/Balanced_Acc"] = (log_dict[f"{log_prefix}/TPR"] + log_dict[f"{log_prefix}/TNR"]) / 2
    do_precision = (confusion_matrix["TP"] + confusion_matrix["FP"]) > 0
    do_recall = (confusion_matrix["TP"] + confusion_matrix["FN"]) > 0
    if do_precision: log_dict[f"{log_prefix}/precision"] = (confusion_matrix["TP"]) / (confusion_matrix["TP"] + confusion_matrix["FP"])
    if do_recall: log_dict[f"{log_prefix}/recall"] = (confusion_matrix["TP"]) / (confusion_matrix["TP"] + confusion_matrix["FN"])
    if do_precision and do_recall:
        log_dict[f"{log_prefix}/f1"] = 0 if confusion_matrix["TP"] == 0 else 2 * (log_dict[f"{log_prefix}/precision"] * log_dict[f"{log_prefix}/recall"]) / (log_dict[f"{log_prefix}/precision"] + log_dict[f"{log_prefix}/recall"])
    return log_dict
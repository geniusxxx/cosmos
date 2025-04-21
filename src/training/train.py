import json
import logging
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.parallel.distributed import DistributedDataParallel

try:
    import wandb
except ImportError:
    wandb = None

from open_clip import get_input_dtype, CLIP, CustomTextCLIP
from .distributed import is_master
from .zero_shot import zero_shot_eval, zero_shot_classification_eval
from .precision import get_autocast

from tqdm import tqdm

from PIL import Image
import csv


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def postprocess_clip_output(model_out):
    return {
        "image_features": model_out[0],
        "text_features": model_out[1],
        "logit_scale": model_out[2]
    }


def unwrap_model(model):
    if hasattr(model, 'module'):
        return model.module
    else:
        return model


def backward(total_loss, scaler):
    if scaler is not None:
        scaler.scale(total_loss).backward()
    else:
        total_loss.backward()


def train_one_epoch(student, teacher, data, loss, epoch, optimizer, scaler, scheduler, momentum_scheduler, dist_model, args, tb_writer=None):
    device = torch.device(args.device)
    autocast = get_autocast(args.precision)
    input_dtype = get_input_dtype(args.precision)

    student.train()
    if args.distill:
        dist_model.eval()

    # set epoch in process safe manner via sampler or shared_epoch
    data['train'].set_epoch(epoch)
    dataloader = data['train'].dataloader
    num_batches_per_epoch = dataloader.num_batches // args.accum_freq
    sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))

    if args.accum_freq > 1:
        accum_images, accum_texts, accum_features = [], [], {}

    use_img_aug = args.use_imagecrop_aug
    use_txt_aug = args.num_sampled_captions > 0

    losses_m = {}
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()
    for i, batch in enumerate(dataloader):
        i_accum = i // args.accum_freq
        step = num_batches_per_epoch * epoch + i_accum

        if not args.skip_scheduler:
            scheduler(step)

        images, texts = batch

        if use_img_aug:
            if isinstance(images[0], list):  # webdataset B, 2 + n_local_crop, {(3,224,224) or (3,96,96)}
                assert args.global_crops_number == 2
                global_images = [torch.stack(img[:2], dim=0) for img in images]  # B (2,3,224,224)
                if args.local_crops_number != 0:
                    local_images = [torch.stack(img[2:], dim=0) for img in images]  # B (n_local_crop,3,96,96)
                global_images = torch.stack(global_images, dim=1)  # (2,B,3,224,224)
                images = [img.to(device=device, dtype=input_dtype, non_blocking=True) for img in global_images]

                if args.local_crops_number != 0:
                    local_images = torch.stack(local_images, dim=1)  # (n_local_crop,B,3,96,96)
                    images += [img.to(device=device, dtype=input_dtype, non_blocking=True) for img in local_images]
            else:  # COCO 2, (B,3,224,224)
                # images include 2 global crops and several local crops
                images = [img.to(device=device, dtype=input_dtype, non_blocking=True) for img in images]
            num_images = len(images)
        else:  # without augmentation just 1 image
            images = images.to(
                device=device, dtype=input_dtype, non_blocking=True)

        batch_size = texts.shape[0]
        if use_txt_aug: # Text augmetation based on synthetic caption dataset
            # (B, n, 77) => (n, B, 77) => (n*B, 77)
            texts = texts.permute(1, 0, 2).reshape(-1, texts.shape[-1])  
            num_texts = args.num_sampled_captions   
        texts = texts.to(device=device, non_blocking=True)

        data_time_m.update(time.time() - end)
        optimizer.zero_grad()

        # assert args.accum_freq == 1
        if args.accum_freq == 1:
            with autocast():
                s_model_out = student(images, texts, batch_size)
                logit_scale = s_model_out['logit_scale']
                distill_logit_scale = s_model_out['distill_logit_scale'] if 'distill_logit_scale' in s_model_out else None

                if use_img_aug:
                    images = torch.cat(images[:2])
                else:
                    images = None

                if use_txt_aug:
                    texts = texts[:batch_size*2] # (B*2, 77)
                else:
                    texts = None
  
                t_model_out = teacher(images, texts)

                model_out = {
                    'logit_scale': logit_scale,
                }
                if distill_logit_scale is not None:
                    model_out['distill_logit_scale'] = distill_logit_scale
                if 'logit_bias' in s_model_out:
                    model_out['logit_bias'] = s_model_out['logit_bias']

                # image features
                if args.cosmos:   
                    model_out['s_image_features'] = s_model_out['image_features'].chunk(num_images)
                    model_out['t_image_features'] = t_model_out['image_features'].chunk(2)
                    model_out['s_img_crossmodal_features'] = s_model_out['img_crossmodal_features'].chunk(num_images)
                else:
                    model_out['s_image_features'] = s_model_out['image_features']

                # text features
                if args.cosmos:
                    model_out['s_text_features'] = s_model_out['text_features'].chunk(num_texts)
                    model_out['t_text_features'] = t_model_out['text_features'].chunk(2)  
                    model_out['s_txt_crossmodal_features'] = s_model_out['txt_crossmodal_features'].chunk(num_texts)     
                else:
                    model_out['s_text_features'] = s_model_out['text_features']

                losses = loss(**model_out, output_dict=True)
                total_loss = sum(losses.values())
                losses["loss"] = total_loss

            backward(total_loss, scaler)
        else:
            # 梯度累积模式 - 严格参考OpenCLIP实现
            # 第一阶段：缓存特征（无梯度）
            with torch.no_grad():
                with autocast():
                    # 学生模型前向传播
                    s_model_out = student(images, texts, batch_size)
                    
                    # 移除不需要缓存的标量
                    for scalar_key in ["logit_scale", "distill_logit_scale", "logit_bias"]:
                        if scalar_key in s_model_out:
                            s_model_out.pop(scalar_key, None)
                    
                    # 准备教师模型输入
                    if use_img_aug:
                        t_images = torch.cat(images[:2])
                    else:
                        t_images = None

                    if use_txt_aug:
                        t_texts = texts[:batch_size*2]
                    else:
                        t_texts = None
                    
                    # 教师模型前向传播
                    t_model_out = teacher(t_images, t_texts)
                    
                    # COSMOS模型缓存特征
                    if args.cosmos:
                        # 将特征按需要分块
                        s_image_features = list(s_model_out['image_features'].chunk(num_images))
                        s_img_crossmodal_features = list(s_model_out['img_crossmodal_features'].chunk(num_images))
                        s_text_features = list(s_model_out['text_features'].chunk(num_texts))
                        s_txt_crossmodal_features = list(s_model_out['txt_crossmodal_features'].chunk(num_texts))
                        t_image_features = list(t_model_out['image_features'].chunk(2))
                        t_text_features = list(t_model_out['text_features'].chunk(2))
                        
                        # 缓存所有特征
                        features_dict = {
                            's_image_features': s_image_features,
                            's_img_crossmodal_features': s_img_crossmodal_features,
                            's_text_features': s_text_features,
                            's_txt_crossmodal_features': s_txt_crossmodal_features,
                            't_image_features': t_image_features,
                            't_text_features': t_text_features
                        }
                    else:
                        # 普通CLIP模型特征
                        features_dict = s_model_out
                    
                    # 存储到累积容器
                    if 'features' not in accum_features:
                        accum_features['features'] = []
                    accum_features['features'].append(features_dict)

                # 缓存输入数据
                accum_images.append(images)
                accum_texts.append(texts)
            
            # 如果不是累积周期的最后一步，继续收集下一批
            if ((i + 1) % args.accum_freq) > 0:
                continue
            
            # 第二阶段：对所有累积的批次计算带梯度的前向和反向传播
            optimizer.zero_grad()
            
            # 对每个批次单独计算梯度
            for j in range(args.accum_freq):
                current_images = accum_images[j]
                current_texts = accum_texts[j]
                
                with autocast():
                    # 为当前批次重新计算特征（带梯度）
                    s_model_out = student(current_images, current_texts, batch_size)
                    logit_scale = s_model_out['logit_scale']
                    distill_logit_scale = s_model_out['distill_logit_scale'] if 'distill_logit_scale' in s_model_out else None
                    
                    # 构建COSMOS需要的模型输出
                    if args.cosmos:
                        # 构建不参与累积的标量参数
                        model_out = {'logit_scale': logit_scale}
                        if distill_logit_scale is not None:
                            model_out['distill_logit_scale'] = distill_logit_scale
                        if 'logit_bias' in s_model_out:
                            model_out['logit_bias'] = s_model_out['logit_bias']
                        
                        # 处理当前批次特征（带梯度）
                        current_s_image_features = list(s_model_out['image_features'].chunk(num_images))
                        current_s_img_crossmodal_features = list(s_model_out['img_crossmodal_features'].chunk(num_images))
                        current_s_text_features = list(s_model_out['text_features'].chunk(num_texts))
                        current_s_txt_crossmodal_features = list(s_model_out['txt_crossmodal_features'].chunk(num_texts))
                        
                        # 构建所有批次特征的集合
                        all_s_image_features = []
                        all_s_img_crossmodal_features = []
                        all_s_text_features = []
                        all_s_txt_crossmodal_features = []
                        
                        # 获取教师特征（只需要当前批次）
                        current_batch_features = accum_features['features'][j]
                        t_image_features = current_batch_features['t_image_features']
                        t_text_features = current_batch_features['t_text_features']
                        
                        # 合并特征（当前批次带梯度，其他批次无梯度）
                        for batch_idx in range(args.accum_freq):
                            batch_features = accum_features['features'][batch_idx]
                            
                            if batch_idx == j:  # 当前批次（带梯度）
                                all_s_image_features.extend(current_s_image_features)
                                all_s_img_crossmodal_features.extend(current_s_img_crossmodal_features)
                                all_s_text_features.extend(current_s_text_features)
                                all_s_txt_crossmodal_features.extend(current_s_txt_crossmodal_features)
                            else:  # 其他批次（无梯度）
                                all_s_image_features.extend(batch_features['s_image_features'])
                                all_s_img_crossmodal_features.extend(batch_features['s_img_crossmodal_features'])
                                all_s_text_features.extend(batch_features['s_text_features'])
                                all_s_txt_crossmodal_features.extend(batch_features['s_txt_crossmodal_features'])
                        
                        # 添加所有特征到模型输出
                        model_out['s_image_features'] = all_s_image_features
                        model_out['s_img_crossmodal_features'] = all_s_img_crossmodal_features
                        model_out['s_text_features'] = all_s_text_features
                        model_out['s_txt_crossmodal_features'] = all_s_txt_crossmodal_features
                        
                        # 添加当前批次的教师特征
                        model_out['t_image_features'] = t_image_features
                        model_out['t_text_features'] = t_text_features
                    else:
                        # 非COSMOS模型处理方式
                        model_out = {}
                        model_out["logit_scale"] = logit_scale
                        if distill_logit_scale is not None:
                            model_out["distill_logit_scale"] = distill_logit_scale
                        if 'logit_bias' in s_model_out:
                            model_out["logit_bias"] = s_model_out['logit_bias']
                        
                        # 处理图像和文本特征
                        all_image_features = []
                        all_text_features = []
                        
                        for batch_idx in range(args.accum_freq):
                            if batch_idx == j:  # 当前批次
                                all_image_features.append(s_model_out['image_features'])
                                all_text_features.append(s_model_out['text_features'])
                            else:  # 其他批次
                                features = accum_features['features'][batch_idx]
                                all_image_features.append(features['image_features'])
                                all_text_features.append(features['text_features'])
                        
                        # 合并特征
                        model_out['s_image_features'] = torch.cat(all_image_features)
                        model_out['s_text_features'] = torch.cat(all_text_features)
                    
                    # 计算损失（不缩放损失）
                    losses = loss(**model_out, output_dict=True)
                    total_loss = sum(losses.values())
                    losses["loss"] = total_loss
                
                # 反向传播
                backward(total_loss, scaler)

        # EMA update for the teacher
        if args.fix_momentum:
            momentum = args.momentum_teacher
        else:
            momentum = momentum_scheduler(step)  # momentum parameter
        with torch.no_grad():
            for param_q, param_k in zip(student.parameters(), teacher.parameters()):
                param_k.data.mul_(momentum).add_(
                    (1 - momentum) * param_q.detach().data)

        if scaler is not None:
            if args.horovod:
                optimizer.synchronize()
                scaler.unscale_(optimizer)
                if args.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        student.parameters(), args.grad_clip_norm, norm_type=2.0)
                    torch.nn.utils.clip_grad_norm_(
                        teacher.parameters(), args.grad_clip_norm, norm_type=2.0)
                with optimizer.skip_synchronize():
                    scaler.step(optimizer)
            else:
                if args.grad_clip_norm is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        student.parameters(), args.grad_clip_norm, norm_type=2.0)
                    torch.nn.utils.clip_grad_norm_(
                        teacher.parameters(), args.grad_clip_norm, norm_type=2.0)
                scaler.step(optimizer)
            scaler.update()
        else:
            if args.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    student.parameters(), args.grad_clip_norm, norm_type=2.0)
                torch.nn.utils.clip_grad_norm_(
                    teacher.parameters(), args.grad_clip_norm, norm_type=2.0)
            optimizer.step()

        # reset gradient accum, if enabled
        if args.accum_freq > 1:
            accum_images, accum_texts, accum_features = [], [], {}

        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        with torch.no_grad():
            unwrap_model(student).logit_scale.clamp_(0, math.log(100))
            unwrap_model(teacher).logit_scale.clamp_(0, math.log(100))
            if distill_logit_scale is not None:
                unwrap_model(student).distill_logit_scale.clamp_(0, math.log(100))
                unwrap_model(teacher).distill_logit_scale.clamp_(0, math.log(100))  

        batch_time_m.update(time.time() - end)
        end = time.time()
        batch_count = i_accum + 1
        if is_master(args) and (i_accum % args.log_every_n_steps == 0 or batch_count == num_batches_per_epoch):
            num_samples = batch_count * batch_size * args.accum_freq * args.world_size
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * batch_count / num_batches_per_epoch

            # NOTE loss is coarsely sampled, just master node and per log update
            for key, val in losses.items():
                if key not in losses_m:
                    losses_m[key] = AverageMeter()
                losses_m[key].update(val.item(), batch_size)

            logit_scale_scalar = logit_scale.item()
            if distill_logit_scale is not None:
                distill_logit_scale_scalar = distill_logit_scale.item()
            loss_log = " ".join(
                [
                    f"{loss_name.capitalize()}: {loss_m.val:#.5g} ({loss_m.avg:#.5g})"
                    for loss_name, loss_m in losses_m.items()
                ]
            )
            samples_per_second = args.accum_freq * \
                args.batch_size * args.world_size / batch_time_m.val
            samples_per_second_per_gpu = args.accum_freq * args.batch_size / batch_time_m.val
            if distill_logit_scale is not None:
                logging.info(
                    f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                    f"Data (t): {data_time_m.avg:.3f} "
                    f"Batch (t): {batch_time_m.avg:.3f}, {samples_per_second:#g}/s, {samples_per_second_per_gpu:#g}/s/gpu "
                    f"LR: {optimizer.param_groups[0]['lr']:5f} "
                    f"Momentum: {momentum:5f} "
                    f"Logit Scale: {math.log(logit_scale_scalar):.3f} " 
                    f"Distill Logit Scale: {math.log(distill_logit_scale_scalar):.3f} " + loss_log
                )
            else:
                logging.info(
                    f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                    f"Data (t): {data_time_m.avg:.3f} "
                    f"Batch (t): {batch_time_m.avg:.3f}, {samples_per_second:#g}/s, {samples_per_second_per_gpu:#g}/s/gpu "
                    f"LR: {optimizer.param_groups[0]['lr']:5f} "
                    f"Momentum: {momentum:5f} "
                    f"Logit Scale: {math.log(logit_scale_scalar):.3f} "  + loss_log
                )
            # Save train loss / etc. Using non avg meter values as loggers have their own smoothing
            log_data = {
                "data_time": data_time_m.val,
                "batch_time": batch_time_m.val,
                "samples_per_second": samples_per_second,
                "samples_per_second_per_gpu": samples_per_second_per_gpu,
                "scale": math.log(logit_scale_scalar),
                "lr": optimizer.param_groups[0]["lr"],
                "momentum": momentum
            }
            if distill_logit_scale is not None:
                log_data['distil_scale'] = math.log(distill_logit_scale_scalar)
            log_data.update({name: val.val for name, val in losses_m.items()})

            log_data = {"train/" + name: val for name, val in log_data.items()}

            if tb_writer is not None:
                for name, val in log_data.items():
                    tb_writer.add_scalar(name, val, step)

            if args.wandb:
                assert wandb is not None, 'Please install wandb.'
                log_data['step'] = step  # for backwards compatibility
                wandb.log(log_data, step=step)

            # resetting batch / data time meters per log window
            batch_time_m.reset()
            data_time_m.reset()
    # end for


def evaluate(model, ema_model, key1, key2, data, epoch, args, tb_writer=None, tokenizer=None):
    metrics = {}
    if not is_master(args):
        return metrics
    device = torch.device(args.device)
    model.eval()
    if ema_model is not None:
        ema_model.eval()

    zero_shot_metrics = zero_shot_eval(
        model, data, epoch, args, tokenizer=tokenizer)
    if key1 == '':
        metrics.update(zero_shot_metrics)
    else:
        temp_zero_shot_metrics = {}
        for k, v in zero_shot_metrics.items():
            temp_zero_shot_metrics[f'{key1}_{k}'] = v
        metrics.update(temp_zero_shot_metrics)

    if ema_model is not None:
        zero_shot_metrics = zero_shot_eval(
            ema_model, data, epoch, args, tokenizer=tokenizer)
        temp_zero_shot_metrics = {}
        for k, v in zero_shot_metrics.items():
            temp_zero_shot_metrics[f'{key2}_{k}'] = v
        metrics.update(temp_zero_shot_metrics)

    autocast = get_autocast(args.precision)
    input_dtype = get_input_dtype(args.precision)

    if args.val_frequency and ((epoch % args.val_frequency) == 0 or epoch == args.epochs):
        if 'val' in data:
            val_dataloader = data['val'].dataloader
            metrics = evaluate_on_split(
                key1, model, val_dataloader, args, epoch, metrics, device, input_dtype, autocast)
            if ema_model is not None:
                metrics = evaluate_on_split(
                    key2, ema_model, val_dataloader, args, epoch, metrics, device, input_dtype, autocast)
                
        if 'train_eval' in data:
            train_eval_dataloader = data['train_eval'].dataloader
            if key1 == '':
                metrics = evaluate_on_split(
                    f'train_eval', model, train_eval_dataloader, args, epoch, metrics, device, input_dtype, autocast)
            else:
                metrics = evaluate_on_split(
                    f'{key1}_train_eval', model, train_eval_dataloader, args, epoch, metrics, device, input_dtype, autocast)
            if ema_model is not None:
                metrics = evaluate_on_split(
                    f'{key2}_train_eval', ema_model, train_eval_dataloader, args, epoch, metrics, device, input_dtype, autocast)                

        if 'val_coco' in data: 
            txt_data, img_data, img2txt_dict, txt2img_dict = data['val_coco']
            txt_loader, img_loader = txt_data.dataloader, img_data.dataloader
            metrics = retrieval_on_split(f'{key1}_coco', model, txt_loader, img_loader, img2txt_dict, txt2img_dict,
                                         args, epoch, metrics, device, input_dtype, autocast)
            if ema_model is not None:
                metrics = retrieval_on_split(f'{key2}_coco', ema_model, txt_loader, img_loader, img2txt_dict, txt2img_dict,
                                            args, epoch, metrics, device, input_dtype, autocast)

        if 'val_flickr' in data: 
            txt_data, img_data, img2txt_dict, txt2img_dict = data['val_flickr']
            txt_loader, img_loader = txt_data.dataloader, img_data.dataloader
            metrics = retrieval_on_split(f'{key1}_flickr', model, txt_loader, img_loader, img2txt_dict, txt2img_dict,
                                         args, epoch, metrics, device, input_dtype, autocast)
            if ema_model is not None:
                metrics = retrieval_on_split(f'{key2}_flickr', ema_model, txt_loader, img_loader, img2txt_dict, txt2img_dict,
                                            args, epoch, metrics, device, input_dtype, autocast)

    if not metrics:
        return metrics

    logging.info(
        f"Eval Epoch: {epoch} "
        + "\t".join([f"{k}: {round(v, 4):.4f}" for k, v in metrics.items()])
    )

    log_data = {"val/" + name: val for name, val in metrics.items()}

    if args.save_logs:
        if tb_writer is not None:
            for name, val in log_data.items():
                tb_writer.add_scalar(name, val, epoch)

        with open(os.path.join(args.checkpoint_path, "results.jsonl"), "a+") as f:
            f.write(json.dumps(metrics))
            f.write("\n")

    if args.wandb:
        assert wandb is not None, 'Please install wandb.'
        if 'train' in data:
            dataloader = data['train'].dataloader
            num_batches_per_epoch = dataloader.num_batches // args.accum_freq
            step = num_batches_per_epoch * epoch
        else:
            step = None
        log_data['epoch'] = epoch
        wandb.log(log_data, step=step)

    return metrics


def zeroshot_evaluate_retrieval(model, ema_model, key1, key2, data, epoch, args, tokenizer=None):
    if not is_master(args):
        return
    device = torch.device(args.device)
    if model is not None:
        model.eval()
    if ema_model is not None:
        ema_model.eval()

    autocast = get_autocast(args.precision)
    input_dtype = get_input_dtype(args.precision)

    # MSCOCO retrieval
    txt_data, img_data, img2txt_dict, txt2img_dict = data['val_coco']
    txt_loader, img_loader = txt_data.dataloader, img_data.dataloader
    if model is not None:
        metrics = retrieval_on_split('', model, txt_loader, img_loader, img2txt_dict, txt2img_dict,
                                    args, epoch, {}, device, input_dtype, autocast)
        logging.info(
            f"Zeroshot Eval COCO {key1}: "
            + "\t".join([f"{k}: {round(v, 4):.4f}" for k, v in metrics.items()])
        )
    if ema_model is not None:
        metrics = retrieval_on_split('', ema_model, txt_loader, img_loader, img2txt_dict, txt2img_dict,
                                    args, epoch, {}, device, input_dtype, autocast)
        logging.info(
            f"Zeroshot Eval COCO {key2}: "
            + "\t".join([f"{k}: {round(v, 4):.4f}" for k, v in metrics.items()])
        )


    # Flickr30k retrieval
    txt_data, img_data, img2txt_dict, txt2img_dict = data['val_flickr']
    txt_loader, img_loader = txt_data.dataloader, img_data.dataloader
    if model is not None:
        metrics = retrieval_on_split('', model, txt_loader, img_loader, img2txt_dict, txt2img_dict,
                                    args, epoch, {}, device, input_dtype, autocast)
        logging.info(
            f"Zeroshot Eval Flickr {key1}: "
            + "\t".join([f"{k}: {round(v, 4):.4f}" for k, v in metrics.items()])
        )
    if ema_model is not None:
        metrics = retrieval_on_split('', ema_model, txt_loader, img_loader, img2txt_dict, txt2img_dict,
                                    args, epoch, {}, device, input_dtype, autocast)
        logging.info(
            f"Zeroshot Eval Flickr {key2}: "
            + "\t".join([f"{k}: {round(v, 4):.4f}" for k, v in metrics.items()])
        )


def zeroshot_evaluate_classification(model, ema_model, key1, key2, data, epoch, args, tokenizer=None): # referring to ALIP repo
    if not is_master(args):
        return
    if model is not None:
        model.eval()
    if ema_model is not None:
        ema_model.eval()

    # ImageNet Classification
    if model is not None:
        zero_shot_metrics = zero_shot_eval(
            model, data, epoch, args, tokenizer=tokenizer)
        logging.info(
            f"Zeroshot Eval ImageNet {key1}: "
            + "\t".join([f"{k}: {round(v, 4):.4f}" for k, v in zero_shot_metrics.items()])
        )

    if ema_model is not None:
        zero_shot_metrics = zero_shot_eval(
            ema_model, data, epoch, args, tokenizer=tokenizer)
        logging.info(
            f"Zeroshot Eval ImageNet {key2}: "
            + "\t".join([f"{k}: {round(v, 4):.4f}" for k, v in zero_shot_metrics.items()])
        )

    dataset_templates = json.load(open('./dataloaders/templates.json'))
    dataset_labels = json.load(open('./dataloaders/label.json'))

    for data_name, dataloader in data.items():
        if data_name == "imagenet-val":
            continue
        assert data_name in ["food101", "cifar10", "cifar100", "sun397", "stanford_car", "aircraft", "dtd", "pets", "flowers", "caltech101"]

        if model is not None:
            zero_shot_metrics = zero_shot_classification_eval(
                model, data_name, dataloader, dataset_labels, dataset_templates, epoch, args, tokenizer=tokenizer)
            logging.info(
                f"Zeroshot Eval {data_name} {key1}: "
                + "\t".join([f"{k}: {round(v, 4):.4f}" for k, v in zero_shot_metrics.items()])
            )

        if ema_model is not None:
            zero_shot_metrics = zero_shot_classification_eval(
                ema_model, data_name, dataloader, dataset_labels, dataset_templates, epoch, args, tokenizer=tokenizer)
            logging.info(
                f"Zeroshot Eval {data_name} {key2}: "
                + "\t".join([f"{k}: {round(v, 4):.4f}" for k, v in zero_shot_metrics.items()])
            )


def evaluate_on_split(keyword, model, dataloader, args, epoch, metrics, device, input_dtype, autocast):
    num_samples = 0
    samples_per_val = dataloader.num_samples

    # FIXME this does not scale past small eval datasets
    # all_image_features @ all_text_features will blow up memory and compute very quickly
    cumulative_loss = 0.0
    cumulative_gen_loss = 0.0
    all_image_features, all_text_features = [], []

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            images, texts = batch

            images = images.to(
                device=device, dtype=input_dtype, non_blocking=True)
            texts = texts.to(device=device, non_blocking=True)

            with autocast():
                model_out = model(images, texts)
                logit_scale = model_out["logit_scale"]
                logit_scale = logit_scale.mean()
                # features are accumulated in CPU tensors, otherwise GPU memory exhausted quickly
                # however, system RAM is easily exceeded and compute time becomes problematic
                image_features = model_out["image_features"]
                text_features = model_out["text_features"]

                all_image_features.append(image_features.cpu())
                all_text_features.append(text_features.cpu())

                logits_per_image = logit_scale * image_features @ text_features.t()
                logits_per_text = logits_per_image.t()

                batch_size = images.shape[0]
                labels = torch.arange(batch_size, device=device).long()
                total_loss = (
                    F.cross_entropy(logits_per_image, labels) +
                    F.cross_entropy(logits_per_text, labels)
                ) / 2

                gen_loss = maybe_compute_generative_loss(model_out)

            cumulative_loss += total_loss * batch_size
            num_samples += batch_size
            if is_master(args) and (i % 100) == 0:
                logging.info(
                    f"Eval Epoch {keyword} : {epoch} [{num_samples} / {samples_per_val}]\t"
                    f"Clip Loss {keyword} : {cumulative_loss / num_samples:.6f}\t")

                if gen_loss is not None:
                    cumulative_gen_loss += gen_loss * batch_size
                    logging.info(
                        f"Generative Loss {keyword} : {cumulative_gen_loss / num_samples:.6f}\t")

        val_metrics = get_clip_metrics(
            image_features=torch.cat(all_image_features),
            text_features=torch.cat(all_text_features),
            logit_scale=logit_scale.cpu(),
        )

        if keyword != '':
            temp_val_metrics = {}
            keyword = keyword + '_'
            for k, v in val_metrics.items():
                temp_val_metrics[keyword + k] = v
            val_metrics = temp_val_metrics

        loss = cumulative_loss / num_samples
        if "epoch" in metrics:  # we only need one epoch information
            metrics.update(
                {**val_metrics, f"{keyword}clip_val_loss": loss.item(),
                 f"{keyword}num_samples": num_samples}
            )
        else:
            metrics.update(
                {**val_metrics, f"{keyword}clip_val_loss": loss.item(),
                 f"epoch": epoch, f"{keyword}num_samples": num_samples}
            )

        if gen_loss is not None:
            gen_loss = cumulative_gen_loss / num_samples
            metrics.update({f"{keyword}val_generative_loss": gen_loss.item()})

    return metrics


def retrieval_on_split(keyword, model, txt_loader, img_loader, img2txt_dict, txt2img_dict, args, epoch, metrics, device, input_dtype, autocast):
    num_txt_samples = txt_loader.num_samples
    num_img_samples = img_loader.num_samples
    all_text_features =  []
    all_cap_ids = []

    with torch.no_grad():
        # first loop over the text dataloader to store all text embeddings
        # for i, batch in tqdm(enumerate(txt_loader), total=len(txt_loader), desc="Processing Texts"):
        for i, batch in tqdm(enumerate(txt_loader)):
            texts, cap_id = batch
            texts = texts.to(device=device, non_blocking=True)
            with autocast():
                text_features = model(text = texts)
                text_features = text_features['text_features']
                all_text_features.append(text_features.detach().cpu())  # cpu list of N, each of shape (B, D)
                all_cap_ids.append(cap_id.detach().cpu())
            
        all_text_features_tensor = torch.cat(all_text_features)  # (N, 512)
        cap_ids = torch.cat(all_cap_ids)

        similarity_scores, img_ids = compute_similarity_scores_original_clip(model, img_loader, all_text_features_tensor, device, input_dtype, autocast)
        new_img2txt_dict, new_txt2img_dict = remap_indices(merged_img_ids=img_ids, cap_ids=cap_ids, img2txt_dict=img2txt_dict, txt2img_dict=txt2img_dict)
        retrieval_metrics = compute_retrieval(similarity_scores=similarity_scores, txt2img=new_txt2img_dict, img2txt=new_img2txt_dict)

        if keyword != '':
            temp_retrieval_metrics = {}
            keyword = keyword + '_'
            for k, v in retrieval_metrics.items():
                temp_retrieval_metrics[keyword + k] = v
            retrieval_metrics = temp_retrieval_metrics

        # We cannot have loss in retrieval task
        if "epoch" in metrics:  # we only need one epoch information
            metrics.update(
                {**retrieval_metrics,
                 f"{keyword}num_text_samples": num_txt_samples,
                 f"{keyword}num_image_samples": num_img_samples
                 }
            )
        else:
            metrics.update(
                {**retrieval_metrics,
                 f"epoch": epoch,
                 f"{keyword}num_text_samples": num_txt_samples,
                 f"{keyword}num_image_samples": num_img_samples
                 }
            )

    return metrics


def compute_similarity_scores_original_clip(model, img_loader, all_text_features_tensor, device, input_dtype, autocast):
    all_image_features = []
    all_img_ids = []

    for batch in tqdm(img_loader):
        images, img_id = batch
        images = images.to(device=device, dtype=input_dtype, non_blocking=True)
        all_img_ids.append(img_id.detach().cpu())
        with autocast():
            image_output = model(image=images)
            image_features = image_output['image_features'] 
            logit_scale = image_output['logit_scale']
            all_image_features.append(image_features.detach().cpu())

    
    all_image_features_tensor = torch.cat(all_image_features)
    img_ids = torch.cat(all_img_ids)

    similarity_scores = logit_scale.cpu() * all_image_features_tensor @ all_text_features_tensor.t()
    return similarity_scores, img_ids


def remap_indices(merged_img_ids, cap_ids, img2txt_dict, txt2img_dict):
    """
    params:
    merged_img_ids: tensor of shape (M, D)
    cap_ids: tensor of shape (N) (But the ordering might be random)
    img2txt_dict: dict mapping each img_id to a list of cap_ids
    txt2img_dict: dict mappint each cap_id to an img_id (a list of one element)
    text_features: tensor of shape (N, D)
    """
    # so now ideally the cap_ids should be (0, ...N), so do the text_features
    # step2: re-index the merged_image_ids and re-do the mapping in the dict.
    # As the original image ids might just be random numbers, they don't represent the real ordering.

    img_id_mapping = {old_id.item(): new_idx for new_idx, old_id in enumerate(merged_img_ids)}

    # Update the img2txt_dict and txt2img_dict with new indices
    new_img2txt_dict = {img_id_mapping[img_id]: [cap_id for cap_id in cap_id_list]
                        for img_id, cap_id_list in img2txt_dict.items()}

    new_txt2img_dict = {cap_id: img_id_mapping[txt2img_dict[cap_id][0]]
                        for cap_id in txt2img_dict.keys()}

    return new_img2txt_dict, new_txt2img_dict


def compute_retrieval(similarity_scores, txt2img, img2txt):
    if isinstance(similarity_scores, tuple):
        i2t_similarity_score, t2i_similarity_score = similarity_scores
    else:
        # Otherwise, treat similarity_scores as a single matrix for t2i
        t2i_similarity_score = similarity_scores.t()
        i2t_similarity_score = similarity_scores

    # compute image -> text
    i2t_ranks = torch.zeros(i2t_similarity_score.shape[0])
    for index, score in enumerate(i2t_similarity_score):
        inds = torch.argsort(score, descending=True)
        # Score
        rank = 1e10
        for i in img2txt[index]:
            tmp = torch.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        i2t_ranks[index] = rank

    # Compute metrics
    ir1 = float(len(torch.where(i2t_ranks < 1)[0]) / len(i2t_ranks))
    ir5 = float(len(torch.where(i2t_ranks < 5)[0]) / len(i2t_ranks))
    ir10 = float(len(torch.where(i2t_ranks < 10)[0]) / len(i2t_ranks))
    i2t_report_dict = {
        "image_to_text_R@1": ir1,
        "image_to_text_R@5": ir5,
        "image_to_text_R@10": ir10,
        "image_to_text_mean_rank": float(i2t_ranks.mean().item() + 1),
        "image_to_text_median_rank": float(np.floor(np.median(i2t_ranks.numpy())) + 1)
    }

    # compute text -> image
    t2i_ranks = torch.zeros(t2i_similarity_score.shape[0])
    for index, score in enumerate(t2i_similarity_score):
        inds = torch.argsort(score, descending=True)
        t2i_ranks[index] = torch.where(inds == txt2img[index])[0][0]

    # Compute metrics
    tr1 = float(len(torch.where(t2i_ranks < 1)[0]) / len(t2i_ranks))
    tr5 = float(len(torch.where(t2i_ranks < 5)[0]) / len(t2i_ranks))
    tr10 = float(len(torch.where(t2i_ranks < 10)[0]) / len(t2i_ranks))
    t2i_report_dict = {
        "text_to_image_R@1": tr1,
        "text_to_image_R@5": tr5,
        "text_to_image_R@10": tr10,
        "text_to_image_mean_rank": float(t2i_ranks.mean().item() + 1),
        "text_to_image_median_rank": float(np.floor(np.median(t2i_ranks.numpy())) + 1)
    }
    metrics = {**t2i_report_dict, **i2t_report_dict}

    return metrics


def get_clip_metrics(image_features, text_features, logit_scale):
    metrics = {}
    logits_per_image = (logit_scale * image_features @
                        text_features.t()).detach().cpu()
    logits_per_text = logits_per_image.t().detach().cpu()

    logits = {"image_to_text": logits_per_image,
              "text_to_image": logits_per_text}
    ground_truth = torch.arange(len(text_features)).view(-1, 1)

    for name, logit in logits.items():
        ranking = torch.argsort(logit, descending=True)
        preds = torch.where(ranking == ground_truth)[1]
        preds = preds.detach().cpu().numpy()
        metrics[f"{name}_mean_rank"] = float(preds.mean() + 1)
        metrics[f"{name}_median_rank"] = float(np.floor(np.median(preds)) + 1)
        for k in [1, 5, 10]:
            metrics[f"{name}_R@{k}"] = float(np.mean(preds < k))

    return metrics


def maybe_compute_generative_loss(model_out):
    if "logits" in model_out and "labels" in model_out:
        token_logits = model_out["logits"]
        token_labels = model_out["labels"]
        return F.cross_entropy(token_logits.permute(0, 2, 1), token_labels)
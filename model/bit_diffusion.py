import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from tqdm import tqdm
from functools import partial
from inspect import isfunction
from einops import repeat
from collections import namedtuple

# 辅助函数
def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def cosine_beta_schedule(timesteps, s = 0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

ModelPrediction = namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])

class GaussianBitDiffusion(nn.Module):
    def __init__(
        self,
        model,
        *,
        num_classes,
        timesteps = 1000,
        ddim_timesteps = 50,
        loss_type = 'l2',
        objective = 'pred_x0',
        beta_schedule = 'cosine',
        condition_x0 = False,
        num_offset_bins = 64
    ):
        super().__init__()
        self.model = model
        self.num_classes = num_classes
        self.objective = objective
        self.condition_x0 = condition_x0
        self.num_offset_bins = num_offset_bins
        
        self.num_timesteps = int(timesteps)
        self.timesteps = int(timesteps)

        if beta_schedule == 'linear':
            betas = torch.linspace(0.0001, 0.02, timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)

        self.ddim_timesteps = ddim_timesteps

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))
        register_buffer('posterior_variance', betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_log_variance_clipped', torch.log(self.posterior_variance.clamp(min =1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        self.loss_fn = F.mse_loss if loss_type == 'l2' else F.l1_loss

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x_0):
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x_0) /
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(self, x, pred_x_start_prev, t, obs, stage_masks):
        model_output = self.model(x, t, stage_masks, obs_cond=obs)
        
        pred_action = model_output['action'][-1] if model_output['action'] is not None else None
        pred_offset = model_output['offset'][-1] if model_output['offset'] is not None else None
        pred_act = model_output['actionness'][-1] if model_output['actionness'] is not None else None

        if self.objective == 'pred_noise':
            pred_noise = pred_action
            pred_x_start = self.predict_start_from_noise(x, t, pred_noise)
            if isinstance(stage_masks, list):
                pred_x_start = pred_x_start * stage_masks[-1]
            else:
                pred_x_start = pred_x_start * stage_masks

        elif self.objective == 'pred_x0':
            pred_x_start = pred_action
            pred_noise = self.predict_noise_from_start(x, t, pred_x_start)
            if isinstance(stage_masks, list):
                pred_noise = pred_noise * stage_masks[-1]
            else:
                pred_noise = pred_noise * stage_masks

        return ModelPrediction(pred_noise, pred_x_start), pred_offset, pred_act

    def p_mean_variance(self, x, pred_x_start_prev, t, obs, stage_masks):
        preds, _, _ = self.model_predictions(x, pred_x_start_prev, t, obs, stage_masks)
        pred_x_start = preds.pred_x_start

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=pred_x_start, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance, pred_x_start

    @torch.no_grad()
    def p_sample_ddim(self, x, pred_x_start_prev, t, t_prev, obs, stage_masks, eta=0.):
        preds, pred_offset_logits, pred_act = self.model_predictions(x, pred_x_start_prev, t, obs, stage_masks)
        pred_noise = preds.pred_noise
        pred_x_start = preds.pred_x_start

        alpha = extract(self.alphas_cumprod, t, x.shape)
        alpha_prev = extract(self.alphas_cumprod, t_prev, x.shape)
        sigma = eta * torch.sqrt((1 - alpha_prev) / (1 - alpha) * (1 - alpha / alpha_prev))
        
        pred_x_start = pred_x_start.clamp(-1., 1.)
        mean_pred = pred_x_start * torch.sqrt(alpha_prev) + torch.sqrt(1 - alpha_prev - sigma ** 2) * pred_noise
        
        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        noise = torch.randn_like(x)
        
        x_prev = mean_pred + nonzero_mask * sigma * noise
        
        # Offset Binning Decoding
        pred_offset_scalar = None
        if pred_offset_logits is not None:
            bin_idx = torch.argmax(pred_offset_logits, dim=-1)
            pred_offset_scalar = (bin_idx.float() + 0.5) / self.num_offset_bins
            pred_offset_scalar = pred_offset_scalar.unsqueeze(-1)
        
        return x_prev, pred_x_start, pred_offset_scalar, pred_act

    @torch.no_grad()
    def p_sample(self, x, pred_x_start_prev, t, obs, stage_masks):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance, pred_x_start = self.p_mean_variance(x, pred_x_start_prev, t, obs, stage_masks)
        noise = torch.randn_like(x)
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise, pred_x_start

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    # ====================================================
    # 确保 p_losses 缩进正确，它是 GaussianBitDiffusion 的成员方法
    # ====================================================
    def p_losses(self,
                t,
                x_0,
                obs,
                mask_all,
                mask_past,
                offset_target=None,
                actionness_target=None,
                offset_loss_weight=1.0,
                noise=None):
        
        x_start = x_0
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_t = self.q_sample(x_start=x_start, t=t, noise=noise)

        obs_cond = obs  

        model_out = self.model(
            x=x_t,
            t=t,
            stage_masks=mask_all,
            obs_cond=obs_cond
        )
        
        pred_action = model_out['action'][-1]
        pred_offset = model_out['offset'][-1] if model_out['offset'] is not None else None
        pred_actionness = model_out['actionness'][-1] if model_out['actionness'] is not None else None

        if self.objective == "pred_noise":
            target_diff = noise
        elif self.objective == "pred_x0":
            target_diff = x_0 
            
        if isinstance(mask_all, list):
            mask_use = mask_all[-1]
        else:
            mask_use = mask_all
            
        loss_diff = self.loss_fn(pred_action, target_diff, reduction="none")
        loss_diff = torch.mean(loss_diff * mask_use)

        total_loss = loss_diff

        # Offset Loss (Classification)
        if pred_offset is not None and offset_target is not None:
            # 1. 制作 Label: float [0, 1] -> bin index [0, 63]
            # 注意: offset_target 是 [-1, 1] 还是 [0, 1] 取决于 futr.py 的处理。
            # 在 futr.py 我看到您之前有 x_0 = x_0_cls * 2.0 - 1.0 的逻辑。
            # 这里 offset_target 我们在 futr.py 中通常会映射到 [-1, 1] 归一化。
            # 但为了做 binning，最好是在 [0, 1] 范围内。
            # 修正建议: 让 offset_target 保持在 [0, 1] 范围内传入 (在 futr.py 修正)，或者在这里反向映射。
            # 假设 futr.py 中 offset_target 是 [-1, 1]:
            offset_01 = (offset_target + 1) / 2.0
            target_bin = (offset_01 * self.num_offset_bins).long().clamp(0, self.num_offset_bins - 1)
            target_bin = target_bin.squeeze(-1) 
            
            pred_offset_in = pred_offset.permute(0, 2, 1) # [B, C, T]
            
            loss_off = F.cross_entropy(pred_offset_in, target_bin, reduction="none")
            
            if mask_use.shape[-1] == 1:
                mask_loss = mask_use.squeeze(-1)
            else:
                mask_loss = mask_use[..., 0]
                
            loss_off = torch.mean(loss_off * mask_loss)
            total_loss = total_loss + (loss_off * offset_loss_weight)
            
        if pred_actionness is not None and actionness_target is not None:
            loss_act = F.mse_loss(pred_actionness, actionness_target, reduction="none")
            loss_act = torch.mean(loss_act * mask_use)
            total_loss = total_loss + loss_act 
            
        return {
            "loss": total_loss,
            "action": pred_action,
            "offset": pred_offset,
            "actionness": pred_actionness,
            "x_t": x_t
        }

    def predict(self,
                x_0,
                obs,
                mask_past,
                masks_stages,
                n_samples=1, 
                n_diffusion_steps=50):
        
        B, T, C = x_0.shape
        obs = repeat(obs, 'b t c -> (s b) t c', s=n_samples)
        
        if isinstance(masks_stages, list):
            masks_stages = [repeat(m, 'b t c -> (s b) t c', s=n_samples) for m in masks_stages]
        else:
            masks_stages = repeat(masks_stages, 'b t c -> (s b) t c', s=n_samples)
            
        mask_past = repeat(mask_past, 'b t c -> (s b) t c', s=n_samples)

        x = torch.randn((n_samples * B, T, C), device=x_0.device)
        step_indices = torch.arange(self.timesteps)[::self.timesteps // n_diffusion_steps].flip(0)
        
        pred_x_start_prev = None
        final_offset = None
        final_actionness = None

        for t_idx in tqdm(step_indices, desc='sampling loop time step', leave=False):
            t = torch.full((x.shape[0],), t_idx, device=x.device, dtype=torch.long)
            
            t_prev_idx = t_idx - (self.timesteps // n_diffusion_steps)
            if t_prev_idx < 0: t_prev_idx = 0 
            t_prev = torch.full((x.shape[0],), t_prev_idx, device=x.device, dtype=torch.long)
            
            x, pred_x0, pred_off, pred_act = self.p_sample_ddim(
                x, pred_x_start_prev, t, t_prev, obs, masks_stages, eta=0.
            )
            
            pred_x_start_prev = pred_x0 
            final_offset = pred_off
            final_actionness = pred_act
            
        outputs = [x]
        if final_offset is not None:
            outputs.append(final_offset)
        if final_actionness is not None:
            outputs.append(final_actionness)
            
        final_res = torch.cat(outputs, dim=-1)
        final_res = final_res.view(n_samples, B, T, -1)
        final_res = final_res.permute(0, 1, 3, 2) 
        
        return final_res
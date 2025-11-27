import argparse
import json
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from skimage.io import imsave

from ldm.models.diffusion.sync_dreamer import SyncMultiviewDiffusion, SyncDDIMSampler
from ldm.util import instantiate_from_config, prepare_inputs
from ldm.lora import (
    inject_trainable_lora_extended,
    inject_trainable_lora_extended3d,
    monkeypatch_remove_lora,
    save_lora_weight,
)


def inject_lora(
    model,
    ckpt_fp: str = None,
    rank: int = 12,
    target_replace_module: list[str] = [
        "DepthAttention"
    ],                                                    # inside DepthTransformer: DepthAttention maybe not(and proj_out's first conv2d)
):
    print(f"[INFO] Injecting LoRA from " + (str(ckpt_fp) if ckpt_fp is not None else "scratch"),)
    require_grad_params = []
    lora_params, _ = inject_trainable_lora_extended(
        model,
        target_replace_module=set(target_replace_module),
        r=rank,
        loras=ckpt_fp,
        eval=True,
    )
    return model


def load_model(cfg, ckpt, strict=True):
    config = OmegaConf.load(cfg)
    model = instantiate_from_config(config.model)
    print(f'loading model from {ckpt} ...')
    ckpt = torch.load(ckpt, map_location='cpu')
    model.load_state_dict(ckpt['state_dict'], strict=strict)

    model = model.cuda().eval()
    model = inject_lora(model, ckpt_fp='output/syncdreamer_finetune/lora/lora_1000.ckpt')
    model = model.cuda().eval()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='configs/syncdreamer.yaml')
    parser.add_argument('--ckpt', type=str, default='ckpt/syncdreamer-step80k.ckpt')
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--transform_fp', type=str, default=None)
    parser.add_argument('--elevation', type=float, default=30)

    parser.add_argument('--sample_num', type=int, default=4)
    parser.add_argument('--crop_size', type=int, default=-1)
    parser.add_argument('--cfg_scale', type=float, default=2.0)
    parser.add_argument('--batch_view_num', type=int, default=8)
    parser.add_argument('--seed', type=int, default=6033)

    parser.add_argument('--sampler', type=str, default='ddim')
    parser.add_argument('--sample_steps', type=int, default=50)
    parser.add_argument('--input_idx', type=int, default=0)
    flags = parser.parse_args()

    torch.random.manual_seed(flags.seed)
    np.random.seed(flags.seed)

    model = load_model(flags.cfg, flags.ckpt, strict=True)
    assert isinstance(model, SyncMultiviewDiffusion)
    Path(f'{flags.output}').mkdir(exist_ok=True, parents=True)
    with open(flags.transform_fp, 'r') as f:
        transform = json.load(f)
        elevation = -1 * transform['frames'][flags.input_idx]['latlon'][0]
    # prepare data
    data = prepare_inputs(flags.input, elevation, flags.crop_size)
    for k, v in data.items():
        data[k] = v.unsqueeze(0).cuda()
        data[k] = torch.repeat_interleave(data[k], flags.sample_num, dim=0)

    if flags.sampler == 'ddim':
        sampler = SyncDDIMSampler(model, flags.sample_steps)
    else:
        raise NotImplementedError
    x_sample = model.sample(sampler, data, flags.cfg_scale, flags.batch_view_num)

    B, N, _, H, W = x_sample.shape
    x_sample = (torch.clamp(x_sample, max=1.0, min=-1.0) + 1) * 0.5
    x_sample = x_sample.permute(0, 1, 3, 4, 2).cpu().numpy() * 255
    x_sample = x_sample.astype(np.uint8)

    for bi in range(B):
        output_fn = Path(flags.output) / f'{bi}.png'
        output_fn_8views = Path(flags.output) / f'{bi}_8views.png'
        mv_idx = flags.input_idx * 2 if flags.transform_fp is None else 0
        imsave(output_fn, np.concatenate([x_sample[bi, ni] for ni in range(-mv_idx, N - mv_idx)], 1))
        imsave(output_fn_8views, np.concatenate([x_sample[bi, ni] for ni in range(-mv_idx, N - mv_idx, 2)], 1))


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    main()

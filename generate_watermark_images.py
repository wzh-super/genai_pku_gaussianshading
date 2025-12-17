"""
生成带水印的图像（支持无水印、Gaussian Shading、TreeRing三种方式）
用法: python generate_watermark_images.py --num 10 --output_dir ./output_images

噪声关系：
- 无水印 & TreeRing：同一个seed → 同一个噪声，TreeRing在傅里叶域做变换
- Gaussian Shading：同一个seed → 但根据watermark构造新噪声，所以噪声不同
"""
import argparse
import os
import json
import torch
from datasets import load_dataset
from inverse_stable_diffusion import InversableStableDiffusionPipeline
from diffusers import DPMSolverMultistepScheduler
from watermark import Gaussian_Shading, Gaussian_Shading_chacha
from image_utils import set_random_seed

# 导入TreeRing官方代码（避免与当前目录的optim_utils冲突）
import importlib.util
import copy
import numpy as np
spec = importlib.util.spec_from_file_location("tr_optim_utils", "./tree-ring-watermark/optim_utils.py")
tr_optim_utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tr_optim_utils)
get_watermarking_mask = tr_optim_utils.get_watermarking_mask
circle_mask = tr_optim_utils.circle_mask


def get_watermarking_pattern_float32(pipe, args, device):
    """生成TreeRing水印pattern（在CPU上做FFT避免cuFFT错误）"""
    set_random_seed(args.w_seed)

    # 在CPU上生成float32的随机噪声并做FFT
    gt_init = torch.randn(1, 4, 64, 64, dtype=torch.float32)  # CPU

    if 'ring' in args.w_pattern:
        gt_patch = torch.fft.fftshift(torch.fft.fft2(gt_init), dim=(-1, -2))
        gt_patch_tmp = copy.deepcopy(gt_patch)
        for i in range(args.w_radius, 0, -1):
            tmp_mask = circle_mask(gt_init.shape[-1], r=i)
            tmp_mask = torch.tensor(tmp_mask)
            for j in range(gt_patch.shape[1]):
                gt_patch[:, j, tmp_mask] = gt_patch_tmp[0, j, 0, i].item()
    elif 'rand' in args.w_pattern:
        gt_patch = torch.fft.fftshift(torch.fft.fft2(gt_init), dim=(-1, -2))
        gt_patch[:] = gt_patch[0]
    elif 'zeros' in args.w_pattern:
        gt_patch = torch.fft.fftshift(torch.fft.fft2(gt_init), dim=(-1, -2)) * 0
    elif 'const' in args.w_pattern:
        gt_patch = torch.fft.fftshift(torch.fft.fft2(gt_init), dim=(-1, -2)) * 0
        gt_patch += args.w_pattern_const
    else:
        gt_patch = torch.fft.fftshift(torch.fft.fft2(gt_init), dim=(-1, -2))

    # 移到GPU
    return gt_patch.to(device)


def inject_watermark_float32(init_latents_w, watermarking_mask, gt_patch, args):
    """注入TreeRing水印（在CPU上做FFT避免cuFFT错误）"""
    # 移到CPU做FFT
    init_latents_w_cpu = init_latents_w.to('cpu').to(torch.float32)
    watermarking_mask_cpu = watermarking_mask.to('cpu')
    gt_patch_cpu = gt_patch.to('cpu')

    init_latents_w_fft = torch.fft.fftshift(torch.fft.fft2(init_latents_w_cpu), dim=(-1, -2))

    if args.w_injection == 'complex':
        init_latents_w_fft[watermarking_mask_cpu] = gt_patch_cpu[watermarking_mask_cpu].clone()
    elif args.w_injection == 'seed':
        init_latents_w_cpu[watermarking_mask_cpu] = gt_patch_cpu[watermarking_mask_cpu].clone().real
        return init_latents_w_cpu.to(init_latents_w.device)

    init_latents_w_cpu = torch.fft.ifft2(torch.fft.ifftshift(init_latents_w_fft, dim=(-1, -2))).real

    # 移回原设备
    return init_latents_w_cpu.to(init_latents_w.device)


# ==================== 数据集加载 ====================

def get_dataset(dataset_path):
    """加载prompt数据集"""
    if 'laion' in dataset_path:
        dataset = load_dataset(dataset_path)['train']
        prompt_key = 'TEXT'
    elif 'coco' in dataset_path:
        # COCO数据集
        meta_data_path = os.path.join(dataset_path, 'meta_data.json')
        with open(meta_data_path) as f:
            data = json.load(f)
            dataset = data['annotations']
            prompt_key = 'caption'
    else:
        # Gustavosta/Stable-Diffusion-Prompts
        dataset = load_dataset(dataset_path)['test']
        prompt_key = 'Prompt'
    return dataset, prompt_key


# ==================== 主函数 ====================

def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 加载模型
    scheduler = DPMSolverMultistepScheduler.from_pretrained(args.model_path, subfolder='scheduler')
    pipe = InversableStableDiffusionPipeline.from_pretrained(
        args.model_path,
        scheduler=scheduler,
        torch_dtype=torch.float16,
        revision='fp16',
    )
    pipe.safety_checker = None
    pipe = pipe.to(device)

    # 加载prompt数据集
    dataset, prompt_key = get_dataset(args.dataset_path)
    print(f'Loaded dataset with {len(dataset)} prompts, using key: {prompt_key}')

    # 初始化Gaussian Shading水印类
    if args.chacha:
        gs_watermark = Gaussian_Shading_chacha(args.channel_copy, args.hw_copy, args.fpr, args.user_number)
    else:
        gs_watermark = Gaussian_Shading(args.channel_copy, args.hw_copy, args.fpr, args.user_number)

    # 初始化TreeRing水印pattern（使用float32版本避免cuFFT错误）
    tr_gt_patch = get_watermarking_pattern_float32(pipe, args, device)

    os.makedirs(args.output_dir, exist_ok=True)

    # 保存所有prompt的记录
    prompts_record = []

    for i in range(args.num):
        seed = i + args.seed

        # 从数据集获取prompt
        current_prompt = dataset[i][prompt_key]

        # ========== 1. 从同一个seed采样原始噪声 ==========
        set_random_seed(seed)
        init_latents_original = pipe.get_random_latents()

        # ========== 2. 无水印生成（直接使用原始噪声） ==========
        outputs_no_wm = pipe(
            current_prompt,
            num_images_per_prompt=1,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            height=args.image_length,
            width=args.image_length,
            latents=init_latents_original.clone(),
        )
        image_no_wm = outputs_no_wm.images[0]

        # ========== 3. TreeRing 水印生成（对原始噪声做傅里叶变换） ==========
        # 复制原始噪声
        init_latents_tr = init_latents_original.clone()

        # 获取水印mask（使用官方函数）
        watermarking_mask = get_watermarking_mask(init_latents_tr, args, device)

        # 注入水印（使用float32版本避免cuFFT错误）
        init_latents_tr = inject_watermark_float32(init_latents_tr, watermarking_mask, tr_gt_patch, args)
        init_latents_tr = init_latents_tr.to(torch.float16)

        outputs_tr = pipe(
            current_prompt,
            num_images_per_prompt=1,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            height=args.image_length,
            width=args.image_length,
            latents=init_latents_tr,
        )
        image_tr = outputs_tr.images[0]

        # ========== 4. Gaussian Shading 水印生成（根据watermark构造新噪声） ==========
        set_random_seed(seed)  # 重置种子
        init_latents_gs = gs_watermark.create_watermark_and_return_w()

        outputs_gs = pipe(
            current_prompt,
            num_images_per_prompt=1,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            height=args.image_length,
            width=args.image_length,
            latents=init_latents_gs,
        )
        image_gs = outputs_gs.images[0]

        # ========== 保存图像 ==========
        image_no_wm.save(os.path.join(args.output_dir, f'{i:04d}_no_watermark.png'))
        image_tr.save(os.path.join(args.output_dir, f'{i:04d}_tree_ring.png'))
        image_gs.save(os.path.join(args.output_dir, f'{i:04d}_gaussian_shading.png'))

        # 记录prompt
        prompts_record.append({
            'index': i,
            'seed': seed,
            'prompt': current_prompt,
            'no_watermark_image': f'{i:04d}_no_watermark.png',
            'tree_ring_image': f'{i:04d}_tree_ring.png',
            'gaussian_shading_image': f'{i:04d}_gaussian_shading.png',
        })

        print(f'[{i+1}/{args.num}] Prompt: {current_prompt[:50]}...')

        # 保存初始噪声用于对比（可选）
        if args.save_latents:
            torch.save({
                'original': init_latents_original.cpu(),
                'tree_ring': init_latents_tr.cpu(),
                'gaussian_shading': init_latents_gs.cpu(),
                'seed': seed,
                'prompt': current_prompt,
            }, os.path.join(args.output_dir, f'{i:04d}_latents.pt'))

    # 保存prompt记录到JSON文件
    prompts_file = os.path.join(args.output_dir, 'prompts.json')
    with open(prompts_file, 'w', encoding='utf-8') as f:
        json.dump(prompts_record, f, indent=2, ensure_ascii=False)

    print(f'\nDone! Images saved to {args.output_dir}')
    print(f'Prompts saved to {prompts_file}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate watermarked images')
    parser.add_argument('--num', type=int, default=10, help='Number of images to generate')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output_dir', type=str, default='./output_images')
    parser.add_argument('--image_length', type=int, default=512)
    parser.add_argument('--guidance_scale', type=float, default=7.5)
    parser.add_argument('--num_inference_steps', type=int, default=50)

    # Gaussian Shading 参数
    parser.add_argument('--channel_copy', type=int, default=1)
    parser.add_argument('--hw_copy', type=int, default=8)
    parser.add_argument('--fpr', type=float, default=0.000001)
    parser.add_argument('--user_number', type=int, default=1000000)
    parser.add_argument('--chacha', action='store_true')

    # TreeRing 参数（与官方代码保持一致）
    parser.add_argument('--w_seed', type=int, default=999999, help='TreeRing pattern seed')
    parser.add_argument('--w_channel', type=int, default=0, help='TreeRing channel (-1 for all)')
    parser.add_argument('--w_radius', type=int, default=10, help='TreeRing radius in Fourier domain')
    parser.add_argument('--w_pattern', type=str, default='ring', help='TreeRing pattern type')
    parser.add_argument('--w_mask_shape', type=str, default='circle', help='TreeRing mask shape')
    parser.add_argument('--w_injection', type=str, default='complex', help='TreeRing injection method')

    # 其他
    parser.add_argument('--save_latents', action='store_true', help='Save initial latents for comparison')
    parser.add_argument('--model_path', type=str, default='/root/autodl-tmp/Shared/stabilityai/stable-diffusion-2-1-base')
    parser.add_argument('--dataset_path', type=str, default='/root/autodl-tmp/Shared/fid_outputs/coco')

    args = parser.parse_args()
    main(args)

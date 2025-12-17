"""
生成带水印的图像
用法: python generate_watermark_images.py --num 10 --output_dir ./output_images
"""
import argparse
import os
import torch
from datasets import load_dataset
from inverse_stable_diffusion import InversableStableDiffusionPipeline
from diffusers import DPMSolverMultistepScheduler
from watermark import Gaussian_Shading, Gaussian_Shading_chacha
from image_utils import set_random_seed


def get_dataset(dataset_path):
    """加载prompt数据集"""
    if 'laion' in dataset_path:
        dataset = load_dataset(dataset_path)['train']
        prompt_key = 'TEXT'
    elif 'coco' in dataset_path:
        import json
        with open('fid_outputs/coco/meta_data.json') as f:
            dataset = json.load(f)
            dataset = dataset['annotations']
            prompt_key = 'caption'
    else:
        # Gustavosta/Stable-Diffusion-Prompts
        dataset = load_dataset(dataset_path)['test']
        prompt_key = 'Prompt'
    return dataset, prompt_key


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

    # 初始化水印类
    if args.chacha:
        watermark = Gaussian_Shading_chacha(args.channel_copy, args.hw_copy, args.fpr, args.user_number)
    else:
        watermark = Gaussian_Shading(args.channel_copy, args.hw_copy, args.fpr, args.user_number)

    os.makedirs(args.output_dir, exist_ok=True)

    for i in range(args.num):
        seed = i + args.seed

        # 从数据集获取prompt
        current_prompt = dataset[i][prompt_key]

        set_random_seed(seed)

        # ========== 1. 无水印生成 ==========
        # 生成标准高斯噪声
        generator = torch.Generator(device=device).manual_seed(seed)
        init_latents_no_wm = torch.randn(1, 4, 64, 64, generator=generator, device=device, dtype=torch.float16)

        outputs_no_wm = pipe(
            current_prompt,
            num_images_per_prompt=1,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            height=args.image_length,
            width=args.image_length,
            latents=init_latents_no_wm,
        )
        image_no_wm = outputs_no_wm.images[0]

        # ========== 2. Gaussian Shading 水印生成 ==========
        set_random_seed(seed)  # 重置种子保证水印生成的随机性一致
        init_latents_wm = watermark.create_watermark_and_return_w()

        outputs_wm = pipe(
            current_prompt,
            num_images_per_prompt=1,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            height=args.image_length,
            width=args.image_length,
            latents=init_latents_wm,
        )
        image_wm = outputs_wm.images[0]

        # ========== 保存图像 ==========
        image_no_wm.save(os.path.join(args.output_dir, f'{i:04d}_no_watermark.png'))
        image_wm.save(os.path.join(args.output_dir, f'{i:04d}_gaussian_shading.png'))

        print(f'[{i+1}/{args.num}] Prompt: {current_prompt[:50]}...')

        # 保存初始噪声和prompt用于对比（可选）
        if args.save_latents:
            torch.save({
                'no_wm': init_latents_no_wm.cpu(),
                'gs_wm': init_latents_wm.cpu(),
                'seed': seed,
                'prompt': current_prompt,
            }, os.path.join(args.output_dir, f'{i:04d}_latents.pt'))

    print(f'\nDone! Images saved to {args.output_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate watermarked images')
    parser.add_argument('--num', type=int, default=10, help='Number of images to generate')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output_dir', type=str, default='./output_images')
    parser.add_argument('--image_length', type=int, default=512)
    parser.add_argument('--guidance_scale', type=float, default=7.5)
    parser.add_argument('--num_inference_steps', type=int, default=50)
    parser.add_argument('--channel_copy', type=int, default=1)
    parser.add_argument('--hw_copy', type=int, default=8)
    parser.add_argument('--fpr', type=float, default=0.000001)
    parser.add_argument('--user_number', type=int, default=1000000)
    parser.add_argument('--chacha', action='store_true')
    parser.add_argument('--save_latents', action='store_true', help='Save initial latents for comparison')
    parser.add_argument('--model_path', type=str, default='/root/autodl-tmp/Shared/stabilityai/stable-diffusion-2-1-base')
    parser.add_argument('--dataset_path', type=str, default='/root/autodl-tmp/Shared/Gustavosta/Stable-Diffusion-Prompts')

    args = parser.parse_args()
    main(args)

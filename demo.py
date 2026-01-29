import argparse
import os
from typing import List

import imageio
import numpy as np
import torch
from diffusers.pipelines import FluxPipeline
from PIL import Image
from huggingface_hub import hf_hub_download

from model.pipeline.flux_refany3d import Condition, generate
from model.train_flux.trainer_refany3d import get_group_mask
from tools.render_glb import render_glb_dr


def load_pipeline(base_model: str, ref_model: str) -> FluxPipeline:
    """Load Flux pipeline and LoRA weights for RefAny3D demo."""
    dtype = torch.bfloat16
    pipe = FluxPipeline.from_pretrained(base_model, torch_dtype=dtype)
    pipe.load_lora_weights(
        ref_model,
        weight_name="coord.safetensors",
        adapter_name="coord",
    )
    pipe.load_lora_weights(
        ref_model,
        weight_name="default.safetensors",
        adapter_name="default",
    )
    pipe.set_adapters(["coord", "default"])

    pipe.transformer.aux_token = True
    pipe.transformer.domain_embedder = torch.nn.Sequential(
        torch.nn.Embedding(num_embeddings=2, embedding_dim=256),
        torch.nn.Linear(256, pipe.transformer.inner_dim, bias=True),
        torch.nn.SiLU(),
        torch.nn.Linear(pipe.transformer.inner_dim, pipe.transformer.inner_dim, bias=True),
    ).to("cuda", dtype=dtype)

    domain_embedder_path = hf_hub_download(repo_id=ref_model, filename="domain_embedder.pth")
    pipe.transformer.domain_embedder.load_state_dict(torch.load(domain_embedder_path))
    pipe = pipe.to("cuda")

    return pipe


def load_conditions(
    video: np.ndarray,
    pointmap_video: np.ndarray,
    condition_size: int,
    n_view: int,
) -> List[Condition]:
    """Prepare conditioning images (RGB and pointmap) for multi-view generation."""
    # Horizontal offset for each reference frame (RGB + pointmap)
    position_deltas = [[0, -32] for _ in range(n_view * 2)]

    interval = max(len(video) // n_view, 1)
    ref_frame_id = [i * interval for i in range(n_view)]

    images = [
        Image.fromarray(frame).resize((condition_size, condition_size))
        for frame in [video[id_] for id_ in ref_frame_id]
    ]
    images.extend(
        [
            Image.fromarray(frame).resize((condition_size, condition_size))
            for frame in [pointmap_video[id_] for id_ in ref_frame_id]
        ]
    )

    conditions: List[Condition] = []
    adapters = ["default"] * n_view + ["default+coord"] * n_view
    for image, adapter, position_delta in zip(images, adapters, position_deltas):
        conditions.append(Condition(image, adapter, position_delta))

    return conditions


def main() -> None:
    parser = argparse.ArgumentParser(description="RefAny3D demo")
    parser.add_argument("--prompt", type=str, help="Text prompt")
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Output directory",
    )
    parser.add_argument("--glb_path", type=str, help="3d object glb model path")
    parser.add_argument(
        "--base_model",
        type=str,
        help="Flux base model path",
        default="black-forest-labs/FLUX.1-dev",
    )
    parser.add_argument(
        "--ref_model",
        type=str,
        help="RefAny3D model path",
        default="JudgementH/RefAny3D",
    )
    parser.add_argument("--img_scale", type=float, default=2.0)
    parser.add_argument("--radius", type=float, default=4.0)
    parser.add_argument("--pitch", type=float, default=0.0)
    parser.add_argument("--y_rotation", type=float, default=0.0)
    parser.add_argument("--num_inference_steps", type=int, default=28)
    parser.add_argument("--condition_size", type=int, default=512)
    parser.add_argument("--n_view", type=int, default=8)
    args = parser.parse_args()

    # load glb and render video
    video = render_glb_dr(
        args.glb_path,
        frame_count=60,
        coordinate_map=False,
        radius=args.radius,
        pitch=args.pitch,
        y_rotation=args.y_rotation,
    )
    pointmap_video = render_glb_dr(
        args.glb_path,
        frame_count=60,
        coordinate_map=True,
        radius=args.radius,
        pitch=args.pitch,
        y_rotation=args.y_rotation,
    )
    # save rgb and point map video
    os.makedirs(args.output_dir, exist_ok=True)
    imageio.mimwrite(f"{args.output_dir}/rgb.mp4", video, fps=30)
    imageio.mimwrite(f"{args.output_dir}/pointmap.mp4", pointmap_video, fps=30)

    # load model
    pipe = load_pipeline(args.base_model, args.ref_model)

    # load conditions
    conditions = load_conditions(video, pointmap_video, args.condition_size, args.n_view)

    group_mask = get_group_mask(n_branch=2 * args.n_view + 3, n_condition=args.n_view)

    res = generate(
        pipe,
        prompt=args.prompt,
        conditions=conditions,
        num_inference_steps=args.num_inference_steps,
        group_mask=torch.tensor(group_mask, dtype=torch.bool),
        image_guidance_scale=args.img_scale,
    )
    img = res.images[0]
    coord = res.coords[0]

    img.save(f"{args.output_dir}/output_rgb.png")
    coord.save(f"{args.output_dir}/output_pointmap.png")
    print(f"Output saved to {args.output_dir}/output_rgb.png")
    print(f"Coord  saved to {args.output_dir}/output_pointmap.png")


if __name__ == "__main__":
    main()

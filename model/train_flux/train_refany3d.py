import json
import os
import random

import imageio
import numpy as np
import torch
import torchvision.transforms as T
import wandb
from PIL import Image

from tools import image_tool

from ..pipeline.flux_refany3d import Condition, generate
from .trainer_refany3d import RefAny3DModel, get_config, train


# Dataset
class MyDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        json_file,
        data_root_path,
        size=512,
        center_crop=True,
        t_drop_rate=0.1,
        i_drop_rate=0.1,
        ref_nums=3,
        pe_config: dict = None,
    ):
        super().__init__()

        self.size = size
        self.center_crop = center_crop
        self.i_drop_rate = i_drop_rate
        self.t_drop_rate = t_drop_rate
        self.ref_nums = ref_nums

        with open(json_file) as f:
            self.data = json.load(f)  # list of dict: [{"image_file": "1.png", "text": "A dog"}]
        self.data_root_path = data_root_path

        self.transform = T.Compose(
            [
                T.Resize(self.size, interpolation=T.InterpolationMode.BILINEAR),
                T.ToTensor(),
            ]
        )

        self.pe_config = pe_config

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        json_path = os.path.join(self.data_root_path, item["json"])
        image_path = os.path.join(self.data_root_path, item["image"])
        # mask_path = os.path.join(self.data_root_path, item['mask'])
        # instance_image = os.path.join(self.data_root_path, item['instance_image'])
        # glb_path = os.path.join(self.data_root_path, item['glb'])
        image_video_path = os.path.join(self.data_root_path, item["video"])
        coord_video_path = os.path.join(self.data_root_path, item["coord"])
        pose_coord_path = os.path.join(self.data_root_path, item["pose_coord"])

        with open(json_path) as f:
            json_data = json.load(f)

        text = json_data["text"]
        item = json_data["item"]
        coord_text = f"coordinate map of {item}"

        # read image
        raw_image = Image.open(image_path)
        image = self.transform(raw_image)
        delta_h = image.shape[1] - self.size
        delta_w = image.shape[2] - self.size
        if self.center_crop:
            top = delta_h // 2
            left = delta_w // 2
        else:
            top = np.random.randint(0, delta_h + 1)
            left = np.random.randint(0, delta_w + 1)
        image = T.functional.crop(image, top=top, left=left, height=self.size, width=self.size)

        # load camera pose
        # pose_cam = torch.from_numpy(np.loadtxt(pose_cam_path))
        pose_coord_image = Image.open(pose_coord_path)
        image_coord = self.transform(pose_coord_image)
        image_coord = T.functional.crop(image_coord, top=top, left=left, height=self.size, width=self.size)
        image_coord, image_coord_mask = image_coord[:3,], image_coord[3:,]
        image_coord_mask = image_coord_mask[0] > 0
        # set background to 1
        image_coord[~image_coord_mask[None].repeat(3, 1, 1)] = 1

        # read video and multiview frames
        raw_video = imageio.v2.mimread(image_video_path)
        raw_video = np.array(raw_video)  # f, h, w, 3

        raw_coord_video = imageio.v2.mimread(coord_video_path)
        raw_coord_video = np.array(raw_coord_video)  # f, h, w, 3

        # calculate reference frame indices based on video length and ref_nums
        video_length = len(raw_coord_video)
        if self.ref_nums == 1:
            ref_frame_id = [0]
        else:
            interval = video_length // self.ref_nums
            ref_frame_id = [i * interval for i in range(self.ref_nums)]

        # additional reference frame
        ref_frames = []
        for id_ in ref_frame_id:
            raw_ref_frame = Image.fromarray(raw_video[id_])
            ref_frame = self.transform(raw_ref_frame)
            ref_frames.append(ref_frame)
        ref_frames = torch.stack(ref_frames)

        # additional reference frame coordinate
        ref_frames_coord = []
        for id_ in ref_frame_id:
            raw_ref_frame = Image.fromarray(raw_coord_video[id_])
            ref_frame = self.transform(raw_ref_frame)
            ref_frames_coord.append(ref_frame)
        ref_frames_coord = torch.stack(ref_frames_coord)

        # Randomly drop text or image
        drop_text = random.random() < self.i_drop_rate
        drop_image = random.random() < self.t_drop_rate
        if drop_text:
            text = ""
            coord_text = ""
        if drop_image:
            ref_frames = torch.zeros_like(ref_frames)
            ref_frames_coord = torch.zeros_like(ref_frames_coord)

        rgb_cond_pe_deltas = self.pe_config["rgb_cond"]
        # rgb_output_pe_deltas = self.pe_config["rgb_output"]
        coord_cond_pe_deltas = self.pe_config["coord_cond"]
        coord_output_pe_deltas = self.pe_config["coord_output"]

        data = {
            "image": image,
            "description": text,
            "coord_image": image_coord,
            "coord_image_position_delta": np.array(coord_output_pe_deltas),
            "coord_description": coord_text,
        }

        for i in range(self.ref_nums):
            data[f"condition_{i}"] = ref_frames[i]
            data[f"condition_type_{i}"] = "subject"
            data[f"position_delta_{i}"] = np.array(rgb_cond_pe_deltas)

        for i in range(self.ref_nums):
            coord_idx = i + self.ref_nums  # coordinate conditions start after RGB conditions
            data[f"condition_{coord_idx}"] = ref_frames_coord[i]
            data[f"condition_type_{coord_idx}"] = "subject"
            data[f"position_delta_{coord_idx}"] = np.array(coord_cond_pe_deltas)

        return data


@torch.no_grad()
def test_function(model, save_path, file_name, step=None):
    condition_size = model.training_config["dataset"]["image_size"]
    target_size = model.training_config["dataset"]["image_size"]
    ref_nums = model.model_config["ref_num"]

    # More details about position delta can be found in the documentation.
    rgb_position_deltas = [model.model_config["pe_config"]["rgb_cond"]] * ref_nums
    coord_position_deltas = [model.model_config["pe_config"]["coord_cond"]] * ref_nums
    position_deltas = rgb_position_deltas + coord_position_deltas

    # Set adapters
    adapters = model.adapter_names[3:]
    condition_type = model.training_config["condition_type"]
    test_list = []

    data_root = model.training_config["dataset"]["data_root_path"]
    data_root = f"{data_root}/part2"
    # Test case1 (in-distribution test case)
    video_path = f"{data_root}/hunyuan_video/00000000.mp4"
    video = imageio.v2.mimread(video_path)
    coord_video_path = f"{data_root}/hunyuan_coord/00000000.mp4"
    coord_video = imageio.v2.mimread(coord_video_path)

    video_length = len(video)
    if ref_nums == 1:
        ref_frame_indices = [0]
    else:
        interval = video_length // ref_nums
        ref_frame_indices = [i * interval for i in range(ref_nums)]

    images = [
        Image.fromarray(frame).resize((condition_size, condition_size))
        for frame in [video[i] for i in ref_frame_indices]
    ]
    images.extend(
        [
            Image.fromarray(frame).resize((condition_size, condition_size))
            for frame in [coord_video[i] for i in ref_frame_indices]
        ]
    )

    prompt = "On a sunlit porch, the Pine-Sol Multi-Surface Cleaner sits atop an outdoor table surrounded by lush greenery. The camera angle is a close-up, focusing on the detailed textures and bright label of the bottle. The morning light is clean and crisp, highlighting the dew on nearby leaves. In the background, hints of a garden with colorful flowers can be seen, complemented by the soft chirping of birds, suggesting a tranquil, nature-infused environment."
    gt_image = Image.open(f"{data_root}/image/00000000.png")
    gt_coord = Image.open(f"{data_root}/pose/00000000/scene_coord_map.png")
    conditions = []
    for image, adapter, position_delta in zip(images, adapters, position_deltas):
        condition = Condition(image, adapter, position_delta)
        conditions.append(condition)
    test_list.append((conditions, prompt, images, "test_case_1", gt_image, gt_coord))

    video_path = f"{data_root}/hunyuan_video/00000020.mp4"
    video = imageio.v2.mimread(video_path)
    coord_video_path = f"{data_root}/hunyuan_coord/00000020.mp4"
    coord_video = imageio.v2.mimread(coord_video_path)
    images = [
        Image.fromarray(frame).resize((condition_size, condition_size))
        for frame in [video[i] for i in ref_frame_indices]
    ]
    images.extend(
        [
            Image.fromarray(frame).resize((condition_size, condition_size))
            for frame in [coord_video[i] for i in ref_frame_indices]
        ]
    )

    prompt = "On a bustling city street, the Mrs. Meyer's Clean Day Hand Soap sits on a small outdoor vendor's table, among a variety of artisanal cleaning products. The photo is taken from a slight aerial view, showcasing the lively street market. The afternoon sun illuminates the scene with dynamic shadows from adjacent buildings. The background is filled with people shopping, carrying eco-friendly bags, and the distant blur of passing traffic. A gentle breeze rustles nearby banners, hinting at a pleasant spring day."
    gt_image = Image.open(f"{data_root}/image/00000020.png")
    gt_coord = Image.open(f"{data_root}/pose/00000020/scene_coord_map.png")
    conditions = []
    for image, adapter, position_delta in zip(images, adapters, position_deltas):
        condition = Condition(image, adapter, position_delta)
        conditions.append(condition)
    test_list.append((conditions, prompt, images, "test_case_2", gt_image, gt_coord))

    # Generate images
    os.makedirs(save_path, exist_ok=True)

    wandb_data = {
        "val/input": [],
        "val/gt": [],
        "val/gt_coord": [],
        "val/output_rgb": [],
        "val/output_coord": [],
    }
    for i, (condition, prompt, input_images, test_case_name, gt_image, gt_coord) in enumerate(test_list):
        generator = torch.Generator(device=model.device)
        generator.manual_seed(42)

        res = generate(
            model.flux_pipe,
            prompt=prompt,
            conditions=condition,
            height=target_size,
            width=target_size,
            generator=generator,
            model_config=model.model_config,
            coord_position_deltas=model.model_config["pe_config"]["coord_output"],
            group_mask=model.model_config["group_mask"],
            kv_cache=False,
        )

        # Save output image
        file_path = os.path.join(save_path, f"{file_name}_{condition_type}_{i}.jpg")
        res.images[0].save(file_path)
        res.coords[0].save(file_path.replace(".jpg", "_coord.jpg"))

        # Prepare data for wandb logging
        input_images_grid = image_tool.create_image_grid(input_images, cols=3)
        input_wandb_image = wandb.Image(input_images_grid, caption=f"{test_case_name}_input")
        gt_wandb_image = wandb.Image(gt_image, caption=f"{test_case_name}_gt")
        gt_coord_wandb_image = wandb.Image(gt_coord, caption=f"{test_case_name}_coord_gt")
        output_wandb_image = wandb.Image(res.images[0], caption=f"{test_case_name}_output")
        output_coord_wandb_image = wandb.Image(res.coords[0], caption=f"{test_case_name}_coord_output")
        wandb_data["val/input"].append(input_wandb_image)
        wandb_data["val/gt"].append(gt_wandb_image)
        wandb_data["val/gt_coord"].append(gt_coord_wandb_image)
        wandb_data["val/output_rgb"].append(output_wandb_image)
        wandb_data["val/output_coord"].append(output_coord_wandb_image)

    # Log to wandb
    wandb.log(wandb_data, step=step)


def main():
    # Initialize
    config = get_config()
    training_config = config["train"]
    if torch.cuda.device_count() > 1:
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK")))

    # Initialize the dataset
    dataset = MyDataset(
        json_file=training_config["dataset"]["json_file"],
        data_root_path=training_config["dataset"]["data_root_path"],
        size=training_config["dataset"]["image_size"],
        center_crop=training_config["dataset"]["center_crop"],
        t_drop_rate=training_config["dataset"]["drop_text_prob"],
        i_drop_rate=training_config["dataset"]["drop_image_prob"],
        ref_nums=config["model"]["ref_num"],  # Number of reference frames
        pe_config=config["model"]["pe_config"],
    )

    # Initialize model
    ref_nums = config["model"]["ref_num"]
    adapter_names = [None, None, "coord"] + ["default"] * ref_nums + ["default+coord"] * ref_nums

    trainable_model = RefAny3DModel(
        flux_pipe_id=config["flux_path"],
        lora_path=training_config.get("lora_path", None),
        lora_config=training_config["lora_config"],
        device="cuda",
        dtype=getattr(torch, config["dtype"]),
        optimizer_config=training_config["optimizer"],
        model_config=config.get("model", {}),
        gradient_checkpointing=training_config.get("gradient_checkpointing", False),
        adapter_names=adapter_names,
        aux_token=config["model"]["aux_token"],
    )

    train(dataset, trainable_model, config, test_function)


if __name__ == "__main__":
    main()

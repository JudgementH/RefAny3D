import argparse
import os

import cv2
import imageio
import numpy as np
import nvdiffrast.torch as dr
import torch
import trimesh


def make_mesh_tensors(mesh, device="cuda", max_tex_size=None):
    mesh_tensors = {}
    if (
        isinstance(mesh.visual, trimesh.visual.texture.TextureVisuals)
        and isinstance(mesh.visual.material, trimesh.visual.material.PBRMaterial)
        and mesh.visual.material.baseColorTexture is None
    ):
        color = mesh.visual.material.baseColorFactor[:3]
        img = (np.array(color) * 255).astype(np.uint8)
        print("WARN: mesh doesn't have baseColorTexture, assigning a pure color")
        mesh.visual.vertex_colors = np.tile(np.array(color).reshape(1, 3), (len(mesh.vertices), 1))
        mesh_tensors["vertex_color"] = (
            torch.as_tensor(mesh.visual.vertex_colors[..., :3], device=device, dtype=torch.float) / 255.0
        )
    elif isinstance(mesh.visual, trimesh.visual.texture.TextureVisuals):
        if isinstance(mesh.visual.material, trimesh.visual.material.PBRMaterial):
            if mesh.visual.material.baseColorTexture is not None:
                img = np.array(mesh.visual.material.baseColorTexture.convert("RGB"))
        else:
            img = np.array(mesh.visual.material.image.convert("RGB"))
        img = img[..., :3]
        if max_tex_size is not None:
            max_size = max(img.shape[0], img.shape[1])
            if max_size > max_tex_size:
                scale = 1 / max_size * max_tex_size
                img = cv2.resize(img, fx=scale, fy=scale, dsize=None)
        mesh_tensors["tex"] = torch.as_tensor(img, device=device, dtype=torch.float)[None] / 255.0
        mesh_tensors["uv_idx"] = torch.as_tensor(mesh.faces, device=device, dtype=torch.int)
        uv = torch.as_tensor(mesh.visual.uv, device=device, dtype=torch.float)
        uv[:, 1] = 1 - uv[:, 1]
        mesh_tensors["uv"] = uv
    else:
        if mesh.visual.vertex_colors is None:
            print("WARN: mesh doesn't have vertex_colors, assigning a pure color")
            mesh.visual.vertex_colors = np.tile(np.array([128, 128, 128]).reshape(1, 3), (len(mesh.vertices), 1))
        mesh_tensors["vertex_color"] = (
            torch.as_tensor(mesh.visual.vertex_colors[..., :3], device=device, dtype=torch.float) / 255.0
        )

    mesh_tensors.update(
        {
            "pos": torch.tensor(mesh.vertices, device=device, dtype=torch.float),
            "faces": torch.tensor(mesh.faces, device=device, dtype=torch.int),
            "vnormals": torch.tensor(mesh.vertex_normals, device=device, dtype=torch.float),
        }
    )
    return mesh_tensors


def get_normalize_model_matrix(vertices: torch.Tensor) -> torch.Tensor:
    """
    使用 PyTorch 生成将模型缩放到 [-1, 1] 范围的 4x4 模型变换矩阵

    参数:
        vertices: (N, 3) 的 torch.Tensor，模型顶点坐标

    返回:
        (4, 4) 的模型变换矩阵
    """
    # Step 1: 包围盒 min/max
    device = vertices.device
    min_coords = vertices.min(dim=0).values
    max_coords = vertices.max(dim=0).values

    # Step 2: 中心点
    center = (min_coords + max_coords) / 2.0

    # Step 3: 最大边长 → 缩放到 [-1, 1]
    scale = 2.0 / (max_coords - min_coords).max()

    # Step 4: 平移矩阵 T
    T = torch.eye(4).to(device)
    T[:3, 3] = -center

    # Step 5: 缩放矩阵 S
    S = torch.eye(4).to(device)
    S[0, 0] = S[1, 1] = S[2, 2] = scale

    # Step 6: 最终变换矩阵 M = S @ T
    M = S @ T
    return M


def get_y_rotation_matrix(angle_degrees: float, device="cuda") -> torch.Tensor:
    """
    生成绕y轴旋转的旋转矩阵

    参数:
        angle_degrees: 旋转角度（度），范围0~360
        device: 设备

    返回:
        (3, 3) 的旋转矩阵
    """
    angle_rad = np.radians(angle_degrees)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)

    rotation = torch.zeros((3, 3), device=device, dtype=torch.float)
    rotation[0, 0] = cos_a
    rotation[0, 2] = sin_a
    rotation[1, 1] = 1.0
    rotation[2, 0] = -sin_a
    rotation[2, 2] = cos_a

    return rotation


def render_glb_dr(
    glb_path, resolution=(512, 512), frame_count=100, coordinate_map=False, radius=4.0, pitch=0.0, y_rotation=0.0
):

    glctx = dr.RasterizeCudaContext()
    mesh = trimesh.load(glb_path)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
        # for m in mesh.geometry.values():
        #     mesh = m
        #     break
    mesh_tensors = make_mesh_tensors(mesh)
    # mesh_tensors = normalize_mesh_tensors(mesh_tensors)
    pos = mesh_tensors["pos"]
    vnormals = mesh_tensors["vnormals"]
    pos_idx = mesh_tensors["faces"]
    has_tex = "tex" in mesh_tensors

    normalized_pos = (pos - pos.min(dim=0)[0]) / (pos.max(dim=0)[0] - pos.min(dim=0)[0])

    # 如果启用coordinate map且需要y轴旋转，对normalized_pos应用旋转
    if coordinate_map and y_rotation != 0.0:
        # 将normalized_pos从[0,1]范围转换到[-1,1]范围进行旋转
        normalized_pos_centered = (normalized_pos - 0.5) * 2.0
        rotation_matrix = get_y_rotation_matrix(y_rotation, device=pos.device)
        normalized_pos_rotated = normalized_pos_centered @ rotation_matrix.T
        # 转换回[0,1]范围并clamp确保在有效范围内
        normalized_pos = torch.clamp((normalized_pos_rotated / 2.0) + 0.5, 0.0, 1.0)

    pos_homo = torch.cat([pos, torch.ones((pos.shape[0], 1), device=pos.device, dtype=pos.dtype)], dim=1)

    model_matrix = get_normalize_model_matrix(pos)
    pos_homo = pos_homo @ model_matrix.T

    # Camera setup
    fov = 45  # Field of view in degrees
    aspect_ratio = resolution[0] / resolution[1]
    near, far = 0.1, 100.0

    # Generate frames
    frames = []
    pitch_rad = np.radians(pitch)
    for frame in range(frame_count):
        angle = (frame / frame_count) * 2 * np.pi
        # 计算水平面上的位置
        horizontal_radius = radius * np.cos(pitch_rad)
        x = horizontal_radius * np.cos(angle)
        z = horizontal_radius * np.sin(angle)
        # 根据俯仰角调整y坐标
        y = radius * np.sin(pitch_rad)
        eye = torch.tensor(
            [x, y, z],
            device="cuda",
            dtype=torch.float,
        )
        center = torch.tensor([0.0, 0.0, 0.0], device="cuda", dtype=torch.float)
        up = torch.tensor([0.0, 1.0, 0.0], device="cuda", dtype=torch.float)

        # View matrix
        z = torch.nn.functional.normalize(eye - center, dim=0)
        x = torch.nn.functional.normalize(torch.linalg.cross(up, z), dim=0)
        y = torch.linalg.cross(z, x)
        view = torch.eye(4, device="cuda", dtype=torch.float)
        view[:3, :3] = torch.stack([x, y, z], dim=0)
        view[:3, 3] = -view[:3, :3] @ eye

        # Projection matrix
        tan_half_fov = np.tan(np.radians(fov) / 2)
        proj = torch.zeros((4, 4), device="cuda", dtype=torch.float)
        proj[0, 0] = 1 / (aspect_ratio * tan_half_fov)
        proj[1, 1] = 1 / tan_half_fov
        proj[2, 2] = -(far + near) / (far - near)
        proj[2, 3] = -(2 * far * near) / (far - near)
        proj[3, 2] = -1

        mvp = proj @ view
        # pos_clip = (mvp[:, None] @ pos_homo[None, ..., None])[..., 0]
        pos_clip = pos_homo @ mvp.T

        # Rasterization
        rast_out, _ = dr.rasterize(glctx, pos_clip[None], pos_idx, resolution)
        if coordinate_map:
            color, _ = dr.interpolate(
                normalized_pos,
                rast_out,
                pos_idx,
            )
        else:
            if has_tex:
                texc, _ = dr.interpolate(mesh_tensors["uv"][None], rast_out, pos_idx)
                color = dr.texture(mesh_tensors["tex"], texc, filter_mode="linear")
            else:
                color, _ = dr.interpolate(mesh_tensors["vertex_color"][None], rast_out, pos_idx)

        # Set background to white
        background = torch.ones_like(color)  # White background
        color = torch.where(rast_out[..., 3:] > 0, color, background)
        color = torch.flip(color, dims=[1])

        # Convert to image
        image = color[0].detach().cpu().numpy()
        image = np.clip(image, 0, 1) * 255
        image = image.astype(np.uint8)
        frames.append(image)

    return frames  # Return frames for further processing or saving


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--glb_path",
        type=str,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
    )
    parser.add_argument("--radius", "-r", type=float, default=4.0)
    parser.add_argument("--pitch", "-p", type=float, default=0.0, help="俯仰角（度），0为水平，正值为向上，负值为向下")
    parser.add_argument("--coordinate_map", type=bool, default=False)
    parser.add_argument(
        "--y_rotation", "-y", type=float, default=0.0, help="coordinate map沿y轴旋转角度（度），范围0~360，默认0"
    )
    args = parser.parse_args()

    if args.y_rotation < 0 or args.y_rotation > 360:
        raise ValueError(f"y_rotation必须在0~360度之间，当前值: {args.y_rotation}")

    glb_path = args.glb_path
    output_dir = args.output_dir
    coordinate_map = args.coordinate_map
    radius = args.radius
    pitch = args.pitch
    y_rotation = args.y_rotation

    os.makedirs(output_dir, exist_ok=True)
    output_path = f"{output_dir}/{os.path.basename(glb_path).replace('.glb', '.mp4')}"
    frames = render_glb_dr(
        glb_path, frame_count=60, coordinate_map=coordinate_map, radius=radius, pitch=pitch, y_rotation=y_rotation
    )

    # Save video
    imageio.mimwrite(output_path, frames, fps=30)
    print(f"Render complete: {output_path}")

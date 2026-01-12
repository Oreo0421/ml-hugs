# Code adapted from 3DGS: https://github.com/graphdeco-inria/gaussian-splatting/blob/main/gaussian_renderer/__init__.py
# License from 3DGS: https://github.com/graphdeco-inria/gaussian-splatting/blob/main/LICENSE.md

#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import math
import torch
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer
)

from hugs.utils.spherical_harmonics import SH2RGB
from hugs.utils.rotations import quaternion_to_matrix


def load_gs_data(file_path):

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"GS file not found: {file_path}")

    # Load the .pt file
    gs_data = torch.load(file_path, map_location='cuda')

    # Ensure all tensors are on CUDA
    for key in gs_data:
        if isinstance(gs_data[key], torch.Tensor):
            gs_data[key] = gs_data[key].cuda()

    # Validate the loaded data structure
    required_keys = ['xyz', 'scales', 'rotq', 'shs', 'opacity', 'active_sh_degree']
    for key in required_keys:
        if key not in gs_data:
            raise KeyError(f"Missing required key in GS data: {key}")

    print(f"Loaded GS data from {file_path}:")
    print(f"  xyz: {gs_data['xyz'].shape}")
    print(f"  scales: {gs_data['scales'].shape}")
    print(f"  rotq: {gs_data['rotq'].shape}")
    print(f"  shs: {gs_data['shs'].shape}")
    print(f"  opacity: {gs_data['opacity'].shape}")
    print(f"  active_sh_degree: {gs_data['active_sh_degree']}")

    return gs_data


def save_combined_pt(human_gs_out, scene_gs_out, output_path):
    """Save combined human and scene Gaussian data to a PT file"""
    combined_gs = {
        'xyz': torch.cat([human_gs_out['xyz'], scene_gs_out['xyz']], dim=0),
        'scales': torch.cat([human_gs_out['scales'], scene_gs_out['scales']], dim=0),
        'rotq': torch.cat([human_gs_out['rotq'], scene_gs_out['rotq']], dim=0),
        'shs': torch.cat([human_gs_out['shs'], scene_gs_out['shs']], dim=0),
        'opacity': torch.cat([human_gs_out['opacity'], scene_gs_out['opacity']], dim=0),
        'active_sh_degree': human_gs_out['active_sh_degree'],  # Use human's active_sh_degree
        # Additional metadata
        'human_n_points': human_gs_out['xyz'].shape[0],
        'scene_n_points': scene_gs_out['xyz'].shape[0],
        'total_n_points': human_gs_out['xyz'].shape[0] + scene_gs_out['xyz'].shape[0],
    }

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save the combined data
    torch.save(combined_gs, output_path)

    print(f"\nCombined PT file saved to: {output_path}")
    print(f"Combined data statistics:")
    print(f"  Total points: {combined_gs['total_n_points']}")
    print(f"  Human points: {combined_gs['human_n_points']}")
    print(f"  Scene points: {combined_gs['scene_n_points']}")
    print(f"  Combined xyz shape: {combined_gs['xyz'].shape}")
    print(f"  Combined scales shape: {combined_gs['scales'].shape}")
    print(f"  Combined rotq shape: {combined_gs['rotq'].shape}")
    print(f"  Combined shs shape: {combined_gs['shs'].shape}")
    print(f"  Combined opacity shape: {combined_gs['opacity'].shape}")

    return combined_gs


def save_rendered_image(image_tensor, save_path, show_image=True):

    if len(image_tensor.shape) == 3:
        image_np = image_tensor.permute(1, 2, 0).detach().cpu().numpy()
    else:
        image_np = image_tensor.detach().cpu().numpy()

    # Ensure values are in [0, 1] range
    image_np = np.clip(image_np, 0.0, 1.0)

    # Convert to uint8 for saving
    image_uint8 = (image_np * 255).astype(np.uint8)

    # Save using PIL
    pil_image = Image.fromarray(image_uint8)
    pil_image.save(save_path)
    print(f"Rendered image saved to: {save_path}")

    # Display the image
    if show_image:
        plt.figure(figsize=(10, 8))
        plt.imshow(image_np)
        plt.title(f"Rendered Image - {os.path.basename(save_path)}")
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    return image_np


def print_render_info(render_pkg, render_mode):

    print(f"\n=== Render Info (Mode: {render_mode}) ===")

    # Main rendered image info
    render_img = render_pkg['render']
    print(f"Main render shape: {render_img.shape}")
    print(f"Main render dtype: {render_img.dtype}")
    print(f"Main render range: [{render_img.min():.4f}, {render_img.max():.4f}]")

    # Human image info if available
    if 'human_img' in render_pkg:
        human_img = render_pkg['human_img']
        print(f"Human render shape: {human_img.shape}")
        print(f"Human render range: [{human_img.min():.4f}, {human_img.max():.4f}]")

    # Visibility filter info
    if 'visibility_filter' in render_pkg:
        vis_filter = render_pkg['visibility_filter']
        print(f"Total Gaussians: {len(vis_filter)}")
        print(f"Visible Gaussians: {vis_filter.sum().item()}")

    # Human and scene specific info
    if 'human_visibility_filter' in render_pkg:
        human_vis = render_pkg['human_visibility_filter']
        print(f"Human Gaussians: {len(human_vis)}")
        print(f"Visible Human Gaussians: {human_vis.sum().item()}")

    if 'scene_visibility_filter' in render_pkg:
        scene_vis = render_pkg['scene_visibility_filter']
        print(f"Scene Gaussians: {len(scene_vis)}")
        print(f"Visible Scene Gaussians: {scene_vis.sum().item()}")

    print("=" * 40)


def render_human_scene(
    data,
    human_gs_out,
    scene_gs_out,
    bg_color,
    human_bg_color=None,
    scaling_modifier=1.0,
    render_mode='human_scene',
    render_human_separate=False,
    save_images=True,
    output_path=None,
    output_dir='./rendered_images',
    show_images=True,
):
    if render_mode in ['human_scene', 'scene'] and scene_gs_out is None:
        raise ValueError(f"scene_gs_out is required for render_mode '{render_mode}'")

    if render_mode in ['human_scene', 'human'] and human_gs_out is None:
        raise ValueError(f"human_gs_out is required for render_mode '{render_mode}'")

    # Create output directory if saving images and no specific output path
    if save_images and output_path is None:
        os.makedirs(output_dir, exist_ok=True)

    feats = None
    if render_mode == 'human_scene':
        feats = torch.cat([human_gs_out['shs'], scene_gs_out['shs']], dim=0)
        means3D = torch.cat([human_gs_out['xyz'], scene_gs_out['xyz']], dim=0)
        opacity = torch.cat([human_gs_out['opacity'], scene_gs_out['opacity']], dim=0)
        scales = torch.cat([human_gs_out['scales'], scene_gs_out['scales']], dim=0)
        rotations = torch.cat([human_gs_out['rotq'], scene_gs_out['rotq']], dim=0)
        active_sh_degree = human_gs_out['active_sh_degree']
    elif render_mode == 'human':
        feats = human_gs_out['shs']
        means3D = human_gs_out['xyz']
        opacity = human_gs_out['opacity']
        scales = human_gs_out['scales']
        rotations = human_gs_out['rotq']
        active_sh_degree = human_gs_out['active_sh_degree']
    elif render_mode == 'scene':
        feats = scene_gs_out['shs']
        means3D = scene_gs_out['xyz']
        opacity = scene_gs_out['opacity']
        scales = scene_gs_out['scales']
        rotations = scene_gs_out['rotq']
        active_sh_degree = scene_gs_out['active_sh_degree']
    else:
        raise ValueError(f'Unknown render mode: {render_mode}')

    render_pkg = render(
        means3D=means3D,
        feats=feats,
        opacity=opacity,
        scales=scales,
        rotations=rotations,
        data=data,
        scaling_modifier=scaling_modifier,
        bg_color=bg_color,
        active_sh_degree=active_sh_degree,
    )

    if render_human_separate and render_mode == 'human_scene':
        render_human_pkg = render(
            means3D=human_gs_out['xyz'],
            feats=human_gs_out['shs'],
            opacity=human_gs_out['opacity'],
            scales=human_gs_out['scales'],
            rotations=human_gs_out['rotq'],
            data=data,
            scaling_modifier=scaling_modifier,
            bg_color=human_bg_color if human_bg_color is not None else bg_color,
            active_sh_degree=human_gs_out['active_sh_degree'],
        )
        render_pkg['human_img'] = render_human_pkg['render']
        render_pkg['human_visibility_filter'] = render_human_pkg['visibility_filter']
        render_pkg['human_radii'] = render_human_pkg['radii']

    # Set up visibility filters for different components
    if render_mode == 'human':
        render_pkg['human_visibility_filter'] = render_pkg['visibility_filter']
        render_pkg['human_radii'] = render_pkg['radii']
    elif render_mode == 'human_scene':
        human_n_gs = human_gs_out['xyz'].shape[0]
        scene_n_gs = scene_gs_out['xyz'].shape[0]
        render_pkg['scene_visibility_filter'] = render_pkg['visibility_filter'][human_n_gs:]
        render_pkg['scene_radii'] = render_pkg['radii'][human_n_gs:]
        if not 'human_visibility_filter' in render_pkg.keys():
            render_pkg['human_visibility_filter'] = render_pkg['visibility_filter'][:-scene_n_gs]
            render_pkg['human_radii'] = render_pkg['radii'][:-scene_n_gs]
    elif render_mode == 'scene':
        render_pkg['scene_visibility_filter'] = render_pkg['visibility_filter']
        render_pkg['scene_radii'] = render_pkg['radii']

    # Print render information
    print_render_info(render_pkg, render_mode)

    # Save and display images
    if save_images or show_images:
        # Determine main image path
        if output_path is not None:
            main_img_path = output_path
        else:
            main_img_path = os.path.join(output_dir, f'rendered_{render_mode}.png')

        # Save main rendered image
        save_rendered_image(render_pkg['render'], main_img_path, show_images)

        # Save human image if available
        if 'human_img' in render_pkg:
            if output_path is not None:
                # Create human-only version with same base name
                base_path = os.path.splitext(output_path)[0]
                ext = os.path.splitext(output_path)[1]
                human_img_path = f"{base_path}_human_only{ext}"
            else:
                human_img_path = os.path.join(output_dir, f'rendered_human_only.png')
            save_rendered_image(render_pkg['human_img'], human_img_path, show_images)

    return render_pkg


def render(means3D, feats, opacity, scales, rotations, data, scaling_modifier=1.0, bg_color=None, active_sh_degree=0):
    if bg_color is None:
        bg_color = torch.zeros(3, dtype=torch.float32, device="cuda")

    screenspace_points = torch.zeros_like(means3D, dtype=means3D.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except Exception as e:
        pass

    means2D = screenspace_points

    # Set up rasterization configuration
    tanfovx = math.tan(data['fovx'] * 0.5)
    tanfovy = math.tan(data['fovy'] * 0.5)

    shs, rgb = None, None
    if len(feats.shape) == 2:
        rgb = feats
    else:
        shs = feats

    raster_settings = GaussianRasterizationSettings(
        image_height=int(data['image_height']),
        image_width=int(data['image_width']),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=data['world_view_transform'],
        projmatrix=data['full_proj_transform'],
        sh_degree=active_sh_degree,
        campos=data['camera_center'],
        prefiltered=False,
        debug=False,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    rendered_image, radii = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        colors_precomp=rgb,
    )
    rendered_image = torch.clamp(rendered_image, 0.0, 1.0)

    return {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter" : radii > 0,
        "radii": radii,
    }


def create_dummy_data():

    data = {
        'fovx': math.pi / 3,  # 60 degrees
        'fovy': math.pi / 3,  # 60 degrees
        'image_height': 512,
        'image_width': 512,
        'world_view_transform': torch.eye(4, dtype=torch.float32, device='cuda'),
        'full_proj_transform': torch.eye(4, dtype=torch.float32, device='cuda'),
        'camera_center': torch.tensor([0.0, 0.0, 5.0], dtype=torch.float32, device='cuda'),
    }
    return data


def create_dummy_scene_gs():
    """Create dummy scene Gaussian Splatting data for testing"""
    n_points = 1000
    scene_gs = {
        'xyz': torch.randn(n_points, 3, dtype=torch.float32, device='cuda') * 2,
        'scales': torch.ones(n_points, 3, dtype=torch.float32, device='cuda') * 0.1,
        'rotq': torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32, device='cuda').repeat(n_points, 1),
        'shs': torch.randn(n_points, 16, 3, dtype=torch.float32, device='cuda') * 0.1,
        'opacity': torch.ones(n_points, 1, dtype=torch.float32, device='cuda') * 0.8,
        'active_sh_degree': 3,
    }
    return scene_gs


if __name__ == "__main__":
    import argparse
    import json
    parser = argparse.ArgumentParser(description='Gaussian Splatting Renderer')
    parser.add_argument('--render_mode', type=str, default='human_scene',
                       choices=['human', 'scene', 'human_scene'],
                       help='Rendering mode')
    parser.add_argument("--camera_json", type=str,
                       help="path to JSON file with camera params")
    parser.add_argument('--human_pt', type=str, default=None,
                       help='Path to human_gs_out.pt file')
    parser.add_argument('--scene_pt', type=str, default=None,
                       help='Path to scene_gs_out.pt file')
    parser.add_argument('--out_png', type=str, default=None,
                       help='Output path for rendered image')
    parser.add_argument('--out_combined_pt', type=str, default=None,
                       help='Output path for combined PT file (only for human_scene mode)')
    parser.add_argument('--output_dir', type=str, default='./rendered_images',
                       help='Output directory for rendered images (used if --out_png not specified)')
    parser.add_argument('--show_images', action='store_true', default=False,
                       help='Show images using matplotlib (requires display)')
    parser.add_argument('--bg_color', type=float, nargs=3, default=[0.0, 0.0, 0.0],
                       help='Background color as RGB values (0-1 range)')

    args = parser.parse_args()

    print("Starting Gaussian Splatting Renderer...")
    print(f"Render mode: {args.render_mode}")
    print(f"Human GS path: {args.human_pt}")
    print(f"Scene GS path: {args.scene_pt}")
    print(f"Output PNG: {args.out_png}")
    print(f"Output combined PT: {args.out_combined_pt}")

    try:
        # Load data based on arguments
        human_gs_out = None
        scene_gs_out = None

        if args.human_pt is not None:
            print(f"Loading human GS data from: {args.human_pt}")
            human_gs_out = load_gs_data(args.human_pt)
        elif args.render_mode in ['human', 'human_scene']:
            print("Creating dummy human GS data (no --human_pt provided)...")
            human_gs_out = create_dummy_scene_gs()

        if args.scene_pt is not None:
            print(f"Loading scene GS data from: {args.scene_pt}")
            scene_gs_out = load_gs_data(args.scene_pt)
        elif args.render_mode in ['scene', 'human_scene']:
            print("Creating dummy scene GS data (no --scene_pt provided)...")
            scene_gs_out = create_dummy_scene_gs()

        shs = scene_gs_out["shs"]          # (N, 16, 3)
        dc = shs[:, 0]                 # (N, 3)
        higher = shs[:, 1:]            # (N, 15, 3)

        print("SH statistics:")
        print("  DC mean abs     :", dc.abs().mean().item())
        print("  Higher mean abs :", higher.abs().mean().item())
        print("  Higher max abs  :", higher.abs().max().item())

        if args.camera_json:
            print(f"Loading camera from JSON: {args.camera_json}")
            with open(args.camera_json, 'r') as f:
                cam_j = json.load(f)
            # pick frame index 0
            i = 0
            # Convert lists → torch tensors on CUDA
            data = {
                'fovx': float(cam_j['fovx'][i]),
                'fovy': float(cam_j['fovy'][i]),
                'image_width': int(cam_j['image_width'][i]),
                'image_height': int(cam_j['image_height'][i]),
                'world_view_transform': torch.tensor(
                    cam_j['world_view_transform'][i],
                    dtype=torch.float32, device='cuda'),
                'full_proj_transform': torch.tensor(
                    cam_j['full_proj_transform'][i],
                    dtype=torch.float32, device='cuda'),
                'camera_center': torch.tensor(
                    cam_j['camera_center'][i],
                    dtype=torch.float32, device='cuda'),
            }
        else:
            print("Creating camera data… (dummy)")
            data = create_dummy_data()

        # Set background color
        bg_color = torch.tensor(args.bg_color, dtype=torch.float32, device='cuda')

        print(f"\nStarting rendering with mode: {args.render_mode}")

        # Render the scene
        render_pkg = render_human_scene(
            data=data,
            human_gs_out=human_gs_out,
            scene_gs_out=scene_gs_out,
            bg_color=bg_color,
            render_mode=args.render_mode,
            render_human_separate=True,
            save_images=True,
            output_path=args.out_png,
            output_dir=args.output_dir,
            show_images=args.show_images,
        )

        # Save combined PT file if in human_scene mode
        if args.render_mode == 'human_scene' and human_gs_out is not None and scene_gs_out is not None:
            if args.out_combined_pt is not None:
                combined_pt_path = args.out_combined_pt
            else:
                # Generate default combined PT file name
                combined_pt_path = os.path.join(args.output_dir, 'combined_human_scene.pt')

            combined_gs = save_combined_pt(human_gs_out, scene_gs_out, combined_pt_path)

        # (1) Histogram of the rendered image
        img_lin = render_pkg["render"].detach().cpu()        # 3×H×W, linear
        print("render range  :", img_lin.min().item(), img_lin.max().item())
        hist_img = torch.histc(img_lin, bins=40, min=0, max=1)
        print("img_hist (40 bins):", hist_img.tolist())

        # (2) Histogram of the DC SH coefficients
        dc = scene_gs_out["shs"][:, 0].reshape(-1).cpu()     # l = 0 term for R,G,B
        hist_dc = torch.histc(dc, bins=40, min=-1, max=1)
        print("dc_hist  (40 bins):", hist_dc.tolist())
        print(f"\nRendering completed successfully!")
        if args.out_png:
            print(f"Main image saved to: {args.out_png}")
        else:
            print(f"Check the '{args.output_dir}' directory for rendered images.")

    except Exception as e:
        print(f"Error during rendering: {e}")
        import traceback
        traceback.print_exc()

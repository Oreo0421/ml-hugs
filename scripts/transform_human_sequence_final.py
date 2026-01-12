#!/usr/bin/env python3
"""
Final Version: Per-frame transform using pose_xx.npz['transl'] or ['trans'],
patched to fallback to zeros if missing, and print transformation summary.
"""

import argparse
from tqdm import tqdm
import torch
import numpy as np
from pathlib import Path
from plyfile import PlyData
from loguru import logger

def load_ply_gaussians(ply_path):
    plydata = PlyData.read(ply_path)
    xyz = np.stack([plydata['vertex'][k] for k in ('x', 'y', 'z')], axis=1)
    opacity = plydata['vertex']['opacity']
    shs = np.stack([plydata['vertex'][f'f_dc_{i}'] for i in range(3)], axis=1)[:, None, :]
    scales = np.exp(np.stack([plydata['vertex'][f'scale_{i}'] for i in range(3)], axis=1))
    rotq = np.stack([plydata['vertex'][f'rot_{i}'] for i in range(4)], axis=1)
    rotq /= np.linalg.norm(rotq, axis=1, keepdims=True)
    return {
        'xyz': xyz,
        'scales': scales,
        'rotq': rotq,
        'shs': shs,
        'opacity': 1 / (1 + np.exp(-opacity[:, None]))
    }

def save_gaussians_to_pt(gaussians, output_path):
    gs_data = {
        'xyz': torch.tensor(gaussians['xyz'], dtype=torch.float32),
        'scales': torch.tensor(gaussians['scales'], dtype=torch.float32),
        'rotq': torch.tensor(gaussians['rotq'], dtype=torch.float32),
        'shs': torch.tensor(gaussians['shs'], dtype=torch.float32),
        'opacity': torch.tensor(gaussians['opacity'], dtype=torch.float32),
        'active_sh_degree': 0
    }
    torch.save(gs_data, output_path)
    logger.info(f"✅ Saved PT: {output_path}")

def apply_transformation_to_gaussians(gaussians, transform_matrix):
    xyz_h = np.concatenate([gaussians['xyz'], np.ones((gaussians['xyz'].shape[0], 1))], axis=1)
    xyz_transformed = (transform_matrix @ xyz_h.T).T[:, :3]
    gaussians['xyz'] = xyz_transformed
    return gaussians

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", "-i", type=str, required=True)
    parser.add_argument("--output_dir", "-o", type=str, required=True)
    parser.add_argument("--output_format", choices=['ply', 'pt', 'both'], default='both')
    parser.add_argument("--start_frame", type=int, default=0)
    parser.add_argument("--end_frame", type=int, default=0)
    parser.add_argument("--pose_npz", type=str, required=True)
    args = parser.parse_args()

    pose_data = np.load(args.pose_npz)
    if "transl" in pose_data:
        transl = pose_data["transl"]
    elif "trans" in pose_data:
        logger.warning("⚠️ Using 'trans' instead of 'transl'")
        transl = pose_data["trans"]
    else:
        logger.warning("⚠️ pose_npz missing 'transl'/'trans' — fallback to zeros")
        transl = np.zeros((args.end_frame + 1, 3))

    transform_matrix_scene = np.array([
        [0.004506839905, -0.124592848122, 0.083404511213, -3.700955867767],
        [0.149711236358, 0.008269036189, 0.004262818955, -2.735711812973],
        [-0.008138610050, 0.083115860820, 0.124601446092, -4.244910240173],
        [0.0, 0.0, 0.0, 1.0]
    ])

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    (output_dir / 'pt').mkdir(parents=True, exist_ok=True)
    (output_dir / 'ply').mkdir(parents=True, exist_ok=True)

    logger.info(f"Processing frames {args.start_frame} to {args.end_frame}")
    logger.info(f"Input dir: {input_dir}")
    logger.info(f"Output dir: {output_dir}")

    latest_transform = None

    for frame_idx in tqdm(range(args.start_frame, args.end_frame + 1), desc="Transforming"):
        input_file = input_dir / f"{frame_idx:08d}.ply"
        if not input_file.exists():
            logger.warning(f"⚠️ Missing: {input_file}")
            continue

        try:
            gaussians = load_ply_gaussians(str(input_file))

            T_pose = np.eye(4)
            T_pose[:3, 3] = transl[frame_idx]
            transform_matrix = transform_matrix_scene @ T_pose
            latest_transform = transform_matrix.copy()

            gaussians_transformed = apply_transformation_to_gaussians(gaussians, transform_matrix)
            gaussians_transformed['xyz'] *= 0.01

            if args.output_format in ['pt', 'both']:
                save_gaussians_to_pt(
                    gaussians_transformed, output_dir / 'pt' / f"{frame_idx:08d}.pt"
                )

            logger.info(f"✅ Frame {frame_idx:08d} done")

        except Exception as e:
            logger.error(f"❌ Frame {frame_idx:08d} failed: {str(e)}")
            continue

    logger.info("🎉 All frames processed successfully!")
    print("\n============================================================")
    print("TRANSFORMATION SUMMARY")
    print("============================================================")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Frames processed: {args.start_frame} to {args.end_frame}")
    print(f"Output format: {args.output_format}")
    print("Transformation matrix applied:")
    print(latest_transform)
    print("============================================================")

if __name__ == "__main__":
    main()

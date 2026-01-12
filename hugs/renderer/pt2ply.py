#!/usr/bin/env python3
"""
Convert Gaussian Splatting PT file to PLY format
Supports both combined human+scene PT files and individual PT files
Optimized for large datasets with vectorized operations
"""

import torch
import numpy as np
import argparse
import os
from pathlib import Path
from plyfile import PlyData, PlyElement


def quaternion_to_rotation_matrix(q):
    """Convert quaternion to rotation matrix - vectorized"""
    # Normalize quaternion
    q = q / torch.norm(q, dim=-1, keepdim=True)

    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

    # Compute rotation matrix elements
    R = torch.zeros((*q.shape[:-1], 3, 3), device=q.device, dtype=q.dtype)

    R[..., 0, 0] = 1 - 2 * (y**2 + z**2)
    R[..., 0, 1] = 2 * (x * y - w * z)
    R[..., 0, 2] = 2 * (x * z + w * y)
    R[..., 1, 0] = 2 * (x * y + w * z)
    R[..., 1, 1] = 1 - 2 * (x**2 + z**2)
    R[..., 1, 2] = 2 * (y * z - w * x)
    R[..., 2, 0] = 2 * (x * z - w * y)
    R[..., 2, 1] = 2 * (y * z + w * x)
    R[..., 2, 2] = 1 - 2 * (x**2 + y**2)

    return R


def compute_covariance_3d_vectorized(scales, rotations):
    """Compute 3D covariance matrix from scales and rotations - fully vectorized"""
    # scales: (N, 3) - scaling factors
    # rotations: (N, 4) - quaternions (w, x, y, z)

    # Create scaling matrix S for all points
    S = torch.diag_embed(scales)  # (N, 3, 3)

    # Convert quaternion to rotation matrix R for all points
    R = quaternion_to_rotation_matrix(rotations)  # (N, 3, 3)

    # Covariance = R * S * S^T * R^T
    RS = torch.bmm(R, S)  # (N, 3, 3)
    covariance = torch.bmm(RS, RS.transpose(-1, -2))  # (N, 3, 3)

    return covariance


def sh_coeffs_to_rgb(shs, active_sh_degree=0):
    """Convert spherical harmonics coefficients to RGB colors - vectorized"""
    # For now, we'll use only the DC component (l=0)
    # The DC component represents the base color

    if len(shs.shape) == 3:  # (N, max_sh_coeffs, 3)
        dc_component = shs[:, 0, :]  # (N, 3) - DC component for RGB
    else:  # (N, 3) - already just RGB
        dc_component = shs

    # Convert from SH DC to RGB (SH DC is scaled by a constant)
    # The DC coefficient is related to RGB by: DC = RGB * sqrt(1/(4*pi))
    C0 = 0.28209479177387814  # sqrt(1/(4*pi))
    rgb = dc_component / C0

    # Add 0.5 to convert from [-0.5, 0.5] to [0, 1] range typically used
    rgb = rgb + 0.5

    # Clamp to valid RGB range
    rgb = torch.clamp(rgb, 0.0, 1.0)

    return rgb


def load_pt_file(pt_path):
    """Load PT file and return Gaussian data"""
    if not os.path.exists(pt_path):
        raise FileNotFoundError(f"PT file not found: {pt_path}")

    print(f"Loading PT file: {pt_path}")
    gs_data = torch.load(pt_path, map_location='cpu')

    # Print information about the loaded data
    print("PT file contents:")
    for key, value in gs_data.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape} {value.dtype}")
        else:
            print(f"  {key}: {value}")

    return gs_data


def convert_pt_to_ply_vectorized(pt_path, ply_path, use_covariance=True, max_sh_degree=0, chunk_size=50000):
    """Convert PT file to PLY format using vectorized operations"""

    # Load PT data
    gs_data = load_pt_file(pt_path)

    # Extract basic data (detach first to handle requires_grad=True)
    xyz = gs_data['xyz'].detach().cpu()  # Keep as tensor for now
    opacity = gs_data['opacity'].detach().cpu().squeeze()  # (N,)
    scales = gs_data['scales'].detach().cpu()  # (N, 3)
    rotations = gs_data['rotq'].detach().cpu()  # (N, 4) - quaternions
    shs = gs_data['shs'].detach().cpu()  # Spherical harmonics
    active_sh_degree = gs_data.get('active_sh_degree', 0)

    n_points = xyz.shape[0]
    print(f"Converting {n_points} Gaussian points to PLY format")
    print(f"Processing in chunks of {chunk_size} points...")

    # Convert SH to RGB colors (vectorized)
    print("Converting spherical harmonics to RGB...")
    rgb = sh_coeffs_to_rgb(shs, active_sh_degree)
    rgb_uint8 = (rgb * 255).clamp(0, 255).byte().numpy()  # (N, 3) uint8

    # Convert basic data to numpy
    xyz_np = xyz.numpy()
    opacity_np = opacity.numpy()
    scales_np = scales.numpy()
    rotations_np = rotations.numpy()

    # Prepare lists to collect all vertex data
    all_vertex_data = []

    # Process in chunks to avoid memory issues
    n_chunks = (n_points + chunk_size - 1) // chunk_size

    for chunk_idx in range(n_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min((chunk_idx + 1) * chunk_size, n_points)
        chunk_n = end_idx - start_idx

        print(f"Processing chunk {chunk_idx + 1}/{n_chunks} ({chunk_n} points)...")

        # Get chunk data
        xyz_chunk = xyz_np[start_idx:end_idx]
        rgb_chunk = rgb_uint8[start_idx:end_idx]
        opacity_chunk = opacity_np[start_idx:end_idx]

        if use_covariance:
            # Compute covariance for this chunk
            scales_chunk = scales[start_idx:end_idx]
            rotations_chunk = rotations[start_idx:end_idx]

            cov_3d = compute_covariance_3d_vectorized(scales_chunk, rotations_chunk)
            cov_np = cov_3d.numpy()  # (chunk_n, 3, 3)

            # Create vertex data for this chunk
            chunk_data = np.column_stack([
                xyz_chunk,  # (chunk_n, 3)
                rgb_chunk,  # (chunk_n, 3)
                opacity_chunk,  # (chunk_n,)
                cov_np[:, 0, 0], cov_np[:, 0, 1], cov_np[:, 0, 2],  # cov_xx, cov_xy, cov_xz
                cov_np[:, 1, 1], cov_np[:, 1, 2],                    # cov_yy, cov_yz
                cov_np[:, 2, 2]                                      # cov_zz
            ])
        else:
            # Use raw scales and rotations
            scales_chunk = scales_np[start_idx:end_idx]
            rotations_chunk = rotations_np[start_idx:end_idx]

            # Create vertex data for this chunk
            chunk_data = np.column_stack([
                xyz_chunk,  # (chunk_n, 3)
                rgb_chunk,  # (chunk_n, 3)
                opacity_chunk,  # (chunk_n,)
                scales_chunk,  # (chunk_n, 3)
                rotations_chunk  # (chunk_n, 4)
            ])

        all_vertex_data.append(chunk_data)

    # Concatenate all chunks
    print("Combining all chunks...")
    vertex_data_array = np.vstack(all_vertex_data)

    # Define the PLY vertex properties
    if use_covariance:
        vertex_dtype = [
            ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
            ('opacity', 'f4'),
            ('cov_xx', 'f4'), ('cov_xy', 'f4'), ('cov_xz', 'f4'),
            ('cov_yy', 'f4'), ('cov_yz', 'f4'), ('cov_zz', 'f4'),
        ]
    else:
        vertex_dtype = [
            ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
            ('opacity', 'f4'),
            ('scale_x', 'f4'), ('scale_y', 'f4'), ('scale_z', 'f4'),
            ('rot_w', 'f4'), ('rot_x', 'f4'), ('rot_y', 'f4'), ('rot_z', 'f4'),
        ]

    # Convert to structured array
    print("Creating structured array for PLY...")
    structured_array = np.zeros(n_points, dtype=vertex_dtype)

    # Fill the structured array
    field_names = [field[0] for field in vertex_dtype]
    for i, field_name in enumerate(field_names):
        structured_array[field_name] = vertex_data_array[:, i]

    # Create vertex element
    print("Creating PLY element...")
    vertex_element = PlyElement.describe(structured_array, 'vertex')

    # Create PLY data and write to file
    print(f"Writing PLY file: {ply_path}")
    ply_data = PlyData([vertex_element])

    # Ensure output directory exists
    os.makedirs(os.path.dirname(ply_path) if os.path.dirname(ply_path) else '.', exist_ok=True)

    # Write PLY file
    ply_data.write(ply_path)

    print(f"Successfully converted PT to PLY: {ply_path}")
    print(f"PLY file contains {n_points} points")

    # Print metadata if available
    if 'human_n_points' in gs_data:
        print(f"  Human points: {gs_data['human_n_points']}")
        print(f"  Scene points: {gs_data['scene_n_points']}")

    return ply_path


def main():
    parser = argparse.ArgumentParser(description='Convert Gaussian Splatting PT file to PLY format')
    parser.add_argument('input_pt', type=str, help='Path to input PT file')
    parser.add_argument('output_ply', type=str, nargs='?', help='Path to output PLY file (optional)')
    parser.add_argument('--use_scales_rotations', action='store_true',
                       help='Use raw scales/rotations instead of computing covariance matrix')
    parser.add_argument('--max_sh_degree', type=int, default=0,
                       help='Maximum spherical harmonics degree to use (default: 0, DC only)')
    parser.add_argument('--chunk_size', type=int, default=50000,
                       help='Number of points to process in each chunk (default: 50000)')

    args = parser.parse_args()

    # Determine output path
    if args.output_ply is None:
        input_path = Path(args.input_pt)
        args.output_ply = str(input_path.with_suffix('.ply'))

    print("PT to PLY Converter (Optimized)")
    print("=" * 50)
    print(f"Input PT:  {args.input_pt}")
    print(f"Output PLY: {args.output_ply}")
    print(f"Use covariance: {not args.use_scales_rotations}")
    print(f"Max SH degree: {args.max_sh_degree}")
    print(f"Chunk size: {args.chunk_size}")
    print("=" * 50)

    try:
        convert_pt_to_ply_vectorized(
            pt_path=args.input_pt,
            ply_path=args.output_ply,
            use_covariance=not args.use_scales_rotations,
            max_sh_degree=args.max_sh_degree,
            chunk_size=args.chunk_size
        )
        print("\nConversion completed successfully!")

    except Exception as e:
        print(f"Error during conversion: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())

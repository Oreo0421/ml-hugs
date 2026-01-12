import numpy as np
import cv2
import argparse
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

def load_joints(npy_path):
    """Load joint data from .npy file"""
    joints = np.load(npy_path)
    print(f"Loaded joints shape: {joints.shape}")
    return joints

def project_3d_to_2d(joints_3d, camera_intrinsics):
    """Project 3D joints to 2D image coordinates"""
    # Convert to homogeneous coordinates
    joints_homo = np.concatenate([joints_3d, np.ones((joints_3d.shape[0], 1))], axis=1)

    # Project to 2D
    joints_2d = []
    for joint in joints_3d:
        # Project point
        p = camera_intrinsics @ joint
        # Normalize by z
        if p[2] > 0:  # Check if point is in front of camera
            x = p[0] / p[2]
            y = p[1] / p[2]
            joints_2d.append([x, y])
        else:
            joints_2d.append([np.nan, np.nan])

    return np.array(joints_2d)

def draw_skeleton(img, joints_2d, connections=None, joint_colors=None, line_color=(0, 255, 0)):
    """Draw joints and skeleton on image"""
    img_copy = img.copy()

    # Default joint colors if not provided
    if joint_colors is None:
        joint_colors = [(255, 0, 0)] * len(joints_2d)  # Red for all joints

    # Draw skeleton connections if provided
    if connections is not None:
        for connection in connections:
            if len(connection) == 2:
                idx1, idx2 = connection
                if idx1 < len(joints_2d) and idx2 < len(joints_2d):
                    pt1 = joints_2d[idx1]
                    pt2 = joints_2d[idx2]
                    if not (np.isnan(pt1).any() or np.isnan(pt2).any()):
                        cv2.line(img_copy,
                                (int(pt1[0]), int(pt1[1])),
                                (int(pt2[0]), int(pt2[1])),
                                line_color, 2)

    # Draw joints
    for i, joint in enumerate(joints_2d):
        if not np.isnan(joint).any():
            x, y = int(joint[0]), int(joint[1])
            # Check if point is within image bounds
            if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
                cv2.circle(img_copy, (x, y), 5, joint_colors[i], -1)
                cv2.circle(img_copy, (x, y), 7, (255, 255, 255), 2)  # White border
                # Add joint index
                cv2.putText(img_copy, str(i), (x+10, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return img_copy

def main():
    parser = argparse.ArgumentParser(description='Visualize 3D joints on RGB image')
    parser.add_argument('--joints', type=str, required=True, help='Path to joints .npy file')
    parser.add_argument('--rgb_image', type=str, required=True, help='Path to RGB image')
    parser.add_argument('--output', type=str, default='joints_visualization.png', help='Output image path')
    args = parser.parse_args()

    # Camera intrinsics from your data
    camera_intrinsics = np.array([
        [1099.97998046875, 0.0, 638.0],
        [0.0, 1099.97998046875, 358.5],
        [0.0, 0.0, 1.0]
    ])

    # Load joints
    joints_3d = load_joints(args.joints)
    print(f"Joint coordinate ranges:")
    print(f"  X: [{joints_3d[:, 0].min():.3f}, {joints_3d[:, 0].max():.3f}]")
    print(f"  Y: [{joints_3d[:, 1].min():.3f}, {joints_3d[:, 1].max():.3f}]")
    print(f"  Z: [{joints_3d[:, 2].min():.3f}, {joints_3d[:, 2].max():.3f}]")

    # Load image
    img = cv2.imread(args.rgb_image)
    if img is None:
        print(f"Error: Could not load image from {args.rgb_image}")
        return

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    print(f"Image shape: {img.shape}")

    # Project 3D joints to 2D
    joints_2d = project_3d_to_2d(joints_3d, camera_intrinsics)

    # Define skeleton connections (common human skeleton - adjust as needed)
    # This is for a typical 22-joint skeleton
    connections = [
        # Torso
        (0, 1), (1, 2), (2, 3), (3, 4),
        # Right arm
        (2, 5), (5, 6), (6, 7), (7, 8),
        # Left arm
        (2, 9), (9, 10), (10, 11), (11, 12),
        # Right leg
        (3, 13), (13, 14), (14, 15), (15, 16),
        # Left leg
        (3, 17), (17, 18), (18, 19), (19, 20),
        # Head
        (1, 21)
    ]

    # Create color scheme for joints
    joint_colors = []
    for i in range(22):
        if i < 5:  # Torso - blue
            joint_colors.append((0, 0, 255))
        elif i < 9:  # Right arm - green
            joint_colors.append((0, 255, 0))
        elif i < 13:  # Left arm - yellow
            joint_colors.append((255, 255, 0))
        elif i < 17:  # Right leg - red
            joint_colors.append((255, 0, 0))
        elif i < 21:  # Left leg - magenta
            joint_colors.append((255, 0, 255))
        else:  # Head - cyan
            joint_colors.append((0, 255, 255))

    # Draw skeleton
    result_img = draw_skeleton(img, joints_2d, connections, joint_colors, line_color=(100, 255, 100))

    # Save and display result
    cv2.imwrite(args.output, result_img)
    print(f"Saved visualization to {args.output}")

    # Display using matplotlib
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))

    # Original image
    ax1.imshow(img_rgb)
    ax1.set_title('Original Image')
    ax1.axis('off')

    # Image with joints
    ax2.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
    ax2.set_title('Image with 3D Joints Projected')
    ax2.axis('off')

    plt.tight_layout()
    plt.savefig('comparison_' + args.output, dpi=150, bbox_inches='tight')
    plt.show()

    # Print joint positions
    print("\n2D Joint positions:")
    for i, joint in enumerate(joints_2d):
        if not np.isnan(joint).any():
            print(f"Joint {i}: ({joint[0]:.1f}, {joint[1]:.1f})")

if __name__ == "__main__":
    main()

#!/usr/bin/env bash

# 用法: ./extract_camera_from_sparse_images.sh your_file.txt new_camera_name.json

SRC_TXT=$(realpath "$1")
NEW_CAM_NAME="$2"
COPY_TARGET_DIR="$3"

TARGET_DIR="/home/fzhi/fzt/3dgs_pipeline/ml_hug/ml-hugs/data/neuman/dataset/lab/sparse/"
MLHUG_DIR="/home/fzhi/fzt/3dgs_pipeline/ml_hug/ml-hugs/"
CAM_DIR="$MLHUG_DIR/output/lab_train_camera_params"



if [ -z "$SRC_TXT" ] || [ -z "$NEW_CAM_NAME" ] || [ -z "$COPY_TARGET_DIR" ]; then
    echo "Usage: $0 your_file.txt new_camera_name.json /path/to/target_dir"
    exit 1
fi

if [ ! -f "$SRC_TXT" ]; then
    echo "Error: file not found: $SRC_TXT"
    exit 1
fi

# 1. 进入 sparse 目录
cd "$TARGET_DIR" || { echo "Cannot cd to $TARGET_DIR"; exit 1; }

# 2. 删除旧的 images.txt
[ -f images.txt ] && rm images.txt

# 3. 复制并命名为 images.txt
cp "$SRC_TXT" images.txt
echo "Copied and renamed to sparse/images.txt"

# 4. 运行相机提取
cd "$MLHUG_DIR" || exit 1
python run_extract_camera.py --seq lab --split train --output output

# 5. 改名 camera 参数文件
cd "$CAM_DIR" || exit 1
if [ ! -f camera_params.json ]; then
    echo "Error: camera_params.json not generated!"
    exit 1
fi

mv camera_params.json "$NEW_CAM_NAME"
echo "Renamed camera_params.json -> $NEW_CAM_NAME"

# 6. 复制到指定目录
mkdir -p "$COPY_TARGET_DIR"
cp "$NEW_CAM_NAME" "$COPY_TARGET_DIR/"
echo "Copied $NEW_CAM_NAME to $COPY_TARGET_DIR"


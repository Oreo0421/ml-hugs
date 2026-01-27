
#set_camera_tx_ty_tz_over_head.py get camera position depending on the tran of human body ply

python set_camera_tx_ty_tz_over_head.py \
  --images_txt images_djr.txt \
  --human_ply /mnt/data_hdd/fzhi/mid/101010/djr/p3/ply/00000000.ply \
  --out_images_txt images_djr_centerp3.txt \
  --image_id 0 \
  --distance 3 \
  --mode center \
  --top_percent 0.5


#generate the camera around the top center

python shift_camera_4dirs_images_txt.py \
  --images_txt images_room_centerp2.txt \
  --out_dir cam_shift_txt \
  --image_id 0 \
  --shift 0.5 \
  --world_up_axis auto \
  --human_ply /mnt/data_hdd/fzhi/mid/101010/room/p2/ply/00000000.ply \
  --auto_axes


#camer extract

./extract_camera_from_sparse_images.sh \
images_room_centerp2_backward_0.50m.txt \
images_room_centerp2_backward.json \
/home/fzhi/fzt/3dgs_pipeline/ml_hug/ml-hugs/camera/

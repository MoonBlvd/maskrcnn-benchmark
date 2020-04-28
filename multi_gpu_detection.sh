python -m torch.distributed.launch \
       --nproc_per_node=4 \
       tools/run_detection.py \
       --config-file configs/a3d/test_mask_rcnn_R_101_FPN_1x_coco.yaml \

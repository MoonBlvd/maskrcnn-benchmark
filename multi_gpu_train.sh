#python -m torch.distributed.launch \
#       --nproc_per_node=3 \
#       tools/train_net.py \
#       --config-file configs/cityscapes/train_mask_rcnn_R_101_FPN_1x_coco2cityscapes.yaml

python -m torch.distributed.launch \
       --nproc_per_node=3 \
       tools/train_net.py \
       --config-file configs/bdd/train_mask_rcnn_R_101_FPN_1x_coco2bdd.yaml



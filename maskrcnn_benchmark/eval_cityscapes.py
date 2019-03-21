from maskrcnn_benchmark.data.datasets.evaluation import evaluate                                          
from maskrcnn_benchmark.data.datasets.cityscapes import CityscapesDataset                                 
import torch
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data import make_data_loader

output_folder='/home/brianyao/Documents/maskrcnn-benchmark/inference/cityscapes_fine_instanceonly_seg_val_cocostyle'                                                                                                
anno_file = '/media/DATA/Cityscapes/annotations/instancesonly_filtered_gtFine_val.json'                   
root = '/media/DATA/Cityscapes/leftImg8bit/val'                                                           
dataset = CityscapesDataset(anno_file, root, True)                                                        

cfg.merge_from_file('../configs/cityscapes/mask_rcnn_coco_eval.yaml') 
cfg.merge_from_list([]) 
cfg.freeze() 

data_loaders_val = make_data_loader(cfg, is_train=False, is_distributed=False) 
data_loader = data_loaders_val[0]


extra_args = dict(
        box_only=False,
        iou_types=("bbox","segm"),
        expected_results=[],
        expected_results_sigma_tol=4,
    )

predictions = torch.load('../inference/cityscapes_fine_instanceonly_seg_val_cocostyle/predictions.pth') 

evaluate(data_loader.dataset, predictions, output_folder,**extra_args)   


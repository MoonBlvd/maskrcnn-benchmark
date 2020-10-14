# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)
from maskrcnn_benchmark.utils.env import setup_environment  # noqa F401 isort:skip

import argparse
import os
import numpy as np
import copy
import glob

import torch
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data import make_data_loader
from maskrcnn_benchmark.engine.inference import inference
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.utils.collect_env import collect_env_info
from maskrcnn_benchmark.utils.comm import synchronize, get_rank
from maskrcnn_benchmark.utils.logger import setup_logger
from maskrcnn_benchmark.utils.miscellaneous import mkdir
from torchvision import transforms as T
from PIL import Image

from maskrcnn_benchmark.utils.comm import all_gather
import pdb
from tqdm import tqdm
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="1, 3"

CATEGORIES = ["__background__ ",
                "person",   # 1
                "car",      # 2
                "rider",    # 3
                "bicycle",  # 4
                "motorcycle",# 5
                "bus",      # 6
                "truck",    # 7
                "train",] 

confidence_threshold = 0.7

BDD_CLASSES = {
        "__background__ ": 0,
        'person': 1,
        'rider': 2,
        'car': 3,
        'bus': 4,
        'truck':5,
        'bike': 6,
        'motor': 7,
        'traffic light': 8,
        'traffic sign': 9,
        'train': 10}

Cityscapes_CLASSES = {
        "__background__ ": 0,
        "person": 1,   # 1
        "car": 2,      # 2
        "rider": 3,    # 3
        "bike": 4,  # 4 bicycle
        "motor": 5,# 5 motorcycle
        "bus": 6,      # 6
        "truck": 7,    # 7
        "train": 8,    # 8
}

City2BDD_id_map = {v:BDD_CLASSES[k] for k, v in Cityscapes_CLASSES.items()}
City2BDD_id_map


def x1y1x2y2_to_cxcywh(bboxes):
    '''
    Added April 14: 
        Bug: previous output was x1y1wh
        Fix: changed to cxcywh
    '''
    bboxes[:,2] -= bboxes[:,0]
    bboxes[:,3] -= bboxes[:,1] 
    bboxes[:,0] += bboxes[:,2]/2
    bboxes[:,1] += bboxes[:,3]/2
    return bboxes

def select_top_predictions(predictions):
    """
    Select only predictions which have a `score` > self.confidence_threshold,
    and returns the predictions in descending order of score

    Arguments:
        predictions (BoxList): the result of the computation by the model.
            It should contain the field `scores`.

    Returns:
        prediction (BoxList): the detected objects. Additional information
            of the detection properties can be found in the fields of
            the BoxList via `prediction.fields()`
    """
    scores = predictions.get_field("scores")
    keep = torch.nonzero(scores > confidence_threshold).squeeze(1)
    predictions = predictions[keep]
    scores = predictions.get_field("scores")
    _, idx = scores.sort(0, descending=True)
    
    return predictions[idx], keep[idx]
    
def build_transform(cfg):
    """
    Creates a basic transformation that was used
    """
    # we are loading images with OpenCV, so we don't need to convert them
    # to BGR, they are already! So all we need to do is to normalize
    # by 255 if we want to convert to BGR255 format, or flip the channels
    # if we want it to be in RGB in [0-1] range.
    if cfg.INPUT.TO_BGR255:
        to_bgr_transform = T.Lambda(lambda x: x * 255)
    else:
        to_bgr_transform = T.Lambda(lambda x: x[[2, 1, 0]])

    normalize_transform = T.Normalize(
        mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD
    )

    transform = T.Compose(
        [
#             T.ToPILImage(),
            T.Resize(cfg.INPUT.MIN_SIZE_TEST),
            T.ToTensor(),
            to_bgr_transform,
            normalize_transform,
        ]
    )
    return transform

def _accumulate_predictions_from_multiple_gpus(predictions_per_gpu):
    all_predictions = all_gather(predictions_per_gpu)
    if not is_main_process():
        return
    # merge the list of dicts
    predictions = {}
    for p in all_predictions:
        predictions.update(p)
    # convert a dict where the key is the index in a list
    image_ids = list(sorted(predictions.keys()))
    if len(image_ids) != image_ids[-1] + 1:
        logger = logging.getLogger("maskrcnn_benchmark.inference")
        logger.warning(
            "Number of images that were gathered from multiple processes is not "
            "a contiguous set. Some images might be missing from the evaluation"
        )

    # convert to a list
    predictions = [predictions[i] for i in image_ids]
    return predictions

def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Inference")
    parser.add_argument(
        "--config-file",
        default="/private/home/fmassa/github/detectron.pytorch_v2/configs/e2e_faster_rcnn_R_50_C4_1x_caffe2.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "-i",
        "--image_folder",
        default="/media/DATA/HEVI_dataset/frames",
        metavar="FILE",
        help="path to the RGB frames",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        default="/media/DATA/HEVI_dataset/detections",
        metavar="FILE",
        help="path to save detection results as numpy",
    )
    parser.add_argument('--gpu', default='0', type=str)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    parser.add_argument("--save_features", type=bool, default=False)

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    distributed = num_gpus > 1
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        synchronize()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    logger = setup_logger("maskrcnn_benchmark", output_dir, get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(cfg)

    logger.info("Collecting env info (might take some time)")
    logger.info("\n" + collect_env_info())

    # initialize model, load checkpointys
    model = build_detection_model(cfg, save_features=args.save_features)
    model.to(cfg.MODEL.DEVICE)
    
    checkpointer = DetectronCheckpointer(cfg, model, save_dir=output_dir)
    _ = checkpointer.load(cfg.MODEL.WEIGHT)
    model.eval()

    # get image transform operator
    transform = build_transform(cfg)
    

    data_loader = make_data_loader(cfg, is_train=False, is_distributed=distributed)
    data_loader = data_loader[0]
    _ = checkpointer.load(cfg.MODEL.WEIGHT)
    
    results_dict = {}
    for i, batch in enumerate(tqdm(data_loader)):
        images, targets, image_ids = batch
        images = images.to(cfg.MODEL.DEVICE)
        with torch.no_grad():
            output = model(images)
            tmp = []
            for j, o in enumerate(output):
                o = o.to('cpu')
                # if convert_pred_coco2cityscapes:
                #     o = coco2cityscapes_label(o)
                output[j] = o
        # results_dict.update(
        #     {img_id: result for img_id, result in zip(image_ids, output)}
        # )

        for o, t in zip(output, targets):
            # if t['video_name'] not in results_dict:
            #     results_dict[t['video_name']] = [o]
            # else:
            #     results_dict[t['video_name']].append(o)
            
            #-------------------
            # NOTE: convert from Cityscapes ID to BDD id
            # labels = o.get_field('labels')
            # for i in range(len(labels)):
            #     labels[i] = City2BDD_id_map[int(labels[i])]
            # o.add_field('labels', labels)   

            # o.resize((1280, 720))
            #-------------------

            save_path = os.path.join(output_dir, t.extra_fields['video_name'])
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            save_path = os.path.join(save_path, str(t.extra_fields['frame_id']).zfill(6)+'.pth')
            torch.save(o, save_path)  

    '''
    # run on all videos
    all_folders = sorted(glob.glob(os.path.join(args.image_folder, '*')))
    for folder in tqdm(all_folders):
        print("Running detection on {}".format(folder))
        video_name = folder.split('/')[-1]    
        output_file = video_name + '_det.npy'
        output_with_feature = []

        all_image_files = sorted(glob.glob(os.path.join(folder, '*.jpg')))
        for frame_id, image_file in enumerate(all_image_files):
            
            # read and transform image
            pil_img = Image.open(image_file)
            img = transform(pil_img)
            img = img.unsqueeze(0).to(cfg.MODEL.DEVICE) # convert to BGR
            
            # run model, select top predictions
            # default output format is (x1, y1, x2, y2)
            if args.save_features:
                predictions, features = model(img)
                predictions = predictions[0].to('cpu')
                features = features.to('cpu').detach()
            else:
                predictions = model(img)
                predictions = predictions[0].to('cpu')

            top_predictions, keep = select_top_predictions(predictions)
            features = features[keep].numpy()

            # get all the outputs to save
            # for late deep-sort tracking purpose, save outputs as
            # [frame_id, track_id(-1), x1, y1, w, h, class, score, features]
            bboxes = top_predictions.bbox.detach().numpy()
            classes = top_predictions.extra_fields['labels'].detach().numpy()
            scores = top_predictions.extra_fields['scores'].detach().numpy()
            frame_ids = frame_id * np.ones([bboxes.shape[0],1])
            track_ids = -1 * np.ones([bboxes.shape[0],1])
            
            # if args.for_deepsort:
            bboxes = x1y1x2y2_to_cxcywh(copy.deepcopy(bboxes))

            classes = np.expand_dims(classes, axis=-1)
            scores = np.expand_dims(scores, axis=-1)
            
            complete_output_array = np.hstack([frame_ids, 
                                               track_ids,
                                               bboxes,
                                               classes,
                                               scores,
                                               features])
            if len(output_with_feature) == 0:
                output_with_feature = complete_output_array
            else:
                output_with_feature = np.vstack([output_with_feature, complete_output_array])
            
            print("Frame id:{}  Number of objects:{}    Classes:{}".format(frame_id, bboxes.shape[0], np.unique(classes)))
        print("Saving detection to {}...".format(os.path.join(args.output_dir, output_file)))
        np.save(os.path.join(args.output_dir, output_file), output_with_feature) 
    '''


if __name__ == "__main__":
    main()

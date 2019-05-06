# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)
from maskrcnn_benchmark.utils.env import setup_environment  # noqa F401 isort:skip

import argparse
import os
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
from maskrcnn_benchmark.data.datasets.evaluation import evaluate


def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Inference")
    parser.add_argument(
        "--config-file",
        default="/private/home/fmassa/github/detectron.pytorch_v2/configs/e2e_faster_rcnn_R_50_C4_1x_caffe2.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    distributed = num_gpus > 1

    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        synchronize()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    save_dir = "."
    logger = setup_logger("maskrcnn_benchmark", save_dir, get_rank(), filename='test_all_BDD_ckpts_log.txt')
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(cfg)

    logger.info("Collecting env info (might take some time)")
    logger.info("\n" + collect_env_info())
    
    # initialize model
    model = build_detection_model(cfg, save_features=False)
    model.to(cfg.MODEL.DEVICE)
    output_dir = cfg.OUTPUT_DIR
    checkpointer = DetectronCheckpointer(cfg, model, save_dir=output_dir)
    # initialize test type, output folders and dataloader
    iou_types = ("bbox",)
    if cfg.MODEL.MASK_ON:
        iou_types = iou_types + ("segm",)
    if cfg.MODEL.KEYPOINT_ON:
        iou_types = iou_types + ("keypoints",)
    output_folders = [None] * len(cfg.DATASETS.TEST)
    dataset_names = cfg.DATASETS.TEST
    if cfg.OUTPUT_DIR:
        for idx, dataset_name in enumerate(dataset_names):
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
            mkdir(output_folder)
            output_folders[idx] = output_folder
    data_loaders_val = make_data_loader(cfg, is_train=False, is_distributed=distributed)

    # Testing on multiple checkpoints if the weight is a directory instead of a .pth file
    if cfg.MODEL.WEIGHT.endswith('.pth') or cfg.MODEL.WEIGHT.endswith('.pkl'):
        all_ckpy_names = [cfg.MODEL.WEIGHT]
    else:
        all_ckpy_names = sorted(glob.glob(os.path.join(cfg.MODEL.WEIGHT,'*.pth')))
    logger.info("Testing on checkpoints:", all_ckpy_names)
    for ckpt_name in all_ckpy_names:
        logger.info("Testing {}".format(ckpt_name))
        _ = checkpointer.load(ckpt_name)#cfg.MODEL.WEIGHT)
        for output_folder, dataset_name, data_loader_val in zip(output_folders, dataset_names, data_loaders_val):
            # if the inference is done, only do the evaluation
            # if os.path.isfile(os.path.join(output_folder, "predictions.pth")):
            #     logger.info("Inference was done, only do evaluation!")
            #     predictions = torch.load(os.path.join(output_folder, "predictions.pth"))
                
            #     extra_args = dict(
            #                     box_only=False if cfg.MODEL.RETINANET_ON else cfg.MODEL.RPN_ONLY,
            #                     iou_types=iou_types,
            #                     expected_results=cfg.TEST.EXPECTED_RESULTS,
            #                     expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
            #                     )   
            #     evaluate(dataset=data_loader_val.dataset,
            #             predictions=predictions,
            #             output_folder=output_folder,
            #             **extra_args)
            # else:
            # logger.info("No inference was done, run inference first")
            inference(
                model,
                data_loader_val,
                dataset_name=dataset_name,
                iou_types=iou_types,
                box_only=False if cfg.MODEL.RETINANET_ON else cfg.MODEL.RPN_ONLY,
                device=cfg.MODEL.DEVICE,
                expected_results=cfg.TEST.EXPECTED_RESULTS,
                expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
                output_folder=output_folder,
                convert_pred_coco2cityscapes=cfg.DATASETS.CONVERT,
            )
            synchronize()


if __name__ == "__main__":
    main()

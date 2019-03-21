# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import datetime
import logging
import time
import os

import torch
from tqdm import tqdm

from maskrcnn_benchmark.data.datasets.evaluation import evaluate
from ..utils.comm import is_main_process, get_world_size
from ..utils.comm import all_gather
from ..utils.comm import synchronize

coco2cityscapes_map = { 1: 1,# person
                        2: 4, # bicycle
                        3: 2, # car
                        4: 5, # motorcycle
                        6: 6, #bus
                        7: 7,# train
                        8: 8,#truck"
                        } 

def coco2cityscapes_label(predictions):
    ''' 
    convert coco class label to cityscapes label.
    Arguments:
        predictions: a BoxList object.
    Return
        predictions: a BoxList object with labels converted.
    '''
    # print('Warning: converting coco model prediction to cityscapes!')
    idx_to_keep = []
    for i in range(len(predictions.extra_fields['labels'])):
        try:
            predictions.extra_fields['labels'][i] = coco2cityscapes_map[int(predictions.extra_fields['labels'][i])]
            idx_to_keep.append(i)
        except:
            '''remove this class since doesnt in Cityscapes'''
            continue

    predictions.bbox  = predictions.bbox[idx_to_keep]
    # print('predictions.extra_fields: ', predictions.extra_fields.keys())
    predictions.extra_fields['scores']  = predictions.extra_fields['scores'][idx_to_keep]
    predictions.extra_fields['mask']  = predictions.extra_fields['mask'][idx_to_keep]
    predictions.extra_fields['labels']  = predictions.extra_fields['labels'][idx_to_keep]
    return predictions


def compute_on_dataset(model, data_loader, device, convert_pred_coco2cityscapes=False):
    model.eval()
    results_dict = {}
    cpu_device = torch.device("cpu")
    for i, batch in enumerate(tqdm(data_loader)):
        images, targets, image_ids = batch
        images = images.to(device)
        with torch.no_grad():
            output = model(images)
            tmp = []
            for j, o in enumerate(output):
                o = o.to(cpu_device)
                if convert_pred_coco2cityscapes:
                    o = coco2cityscapes_label(o)
                output[j] = o
        results_dict.update(
            {img_id: result for img_id, result in zip(image_ids, output)}
        )
    return results_dict


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


def inference(
        model,
        data_loader,
        dataset_name,
        iou_types=("bbox",),
        box_only=False,
        device="cuda",
        expected_results=(),
        expected_results_sigma_tol=4,
        output_folder=None,
        convert_pred_coco2cityscapes=False,
):
    # convert to a torch.device for efficiency
    device = torch.device(device)
    num_devices = get_world_size()
    logger = logging.getLogger("maskrcnn_benchmark.inference")
    dataset = data_loader.dataset
    logger.info("Start evaluation on {} dataset({} images).".format(dataset_name, len(dataset)))
    start_time = time.time()
    predictions = compute_on_dataset(model, data_loader, device, convert_pred_coco2cityscapes)

    # if convert_pred_coco2cityscapes:
    #     '''Only convert it when testing COCO trained model on Cityscpaes dataset'''
    #     print('Warning: converting coco model prediction to cityscapes!')
    #     predictions = coco2cityscapes_label(predictions)

    # wait for all processes to complete before measuring the time
    synchronize()
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=total_time))
    logger.info(
        "Total inference time: {} ({} s / img per device, on {} devices)".format(
            total_time_str, total_time * num_devices / len(dataset), num_devices
        )
    )

    predictions = _accumulate_predictions_from_multiple_gpus(predictions)
    if not is_main_process():
        return

    if output_folder:
        torch.save(predictions, os.path.join(output_folder, "predictions.pth"))

    extra_args = dict(
        box_only=box_only,
        iou_types=iou_types,
        expected_results=expected_results,
        expected_results_sigma_tol=expected_results_sigma_tol,
    )

    return evaluate(dataset=dataset,
                    predictions=predictions,
                    output_folder=output_folder,
                    **extra_args)

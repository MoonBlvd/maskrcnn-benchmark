import cv2
import torch
import numpy as np
import copy
import colorsys
from collections import defaultdict
# from maskrcnn_benchmark.utils.cv2_util import findContours
from maskrcnn_benchmark.structures.image_list import ImageList
from maskrcnn_benchmark.utils import cv2_util
import pdb

class Visualizer():
    def __init__(self, cfg):
        self.cfg = cfg
        # used to make colors for each class
        self.palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
        
    def visualize(self,
                images, 
                bboxes, 
                dataset, 
                score_thresh=0.25,
                viz_2D=False, 
                viz_pred_2D=False,
                draw_gt=True):
        '''
        July 11
        Brian Yao
        Draw visualization while training
        Params:
            images: ImageList or tensor (1, 3, H, W)
            bboxes: BoxList (eval)
            dataset: dataset object
        '''
        plots = {}

        # get rgb image 
        if isinstance(images, ImageList):
            images = images.tensors[0]
        rgb = images.permute(1,2,0).detach().cpu().numpy() 
        rgb = rgb * np.array(self.cfg.INPUT.PIXEL_STD) + np.array(self.cfg.INPUT.PIXEL_MEAN)
        rgb = cv2.cvtColor(rgb.astype('uint8'), cv2.COLOR_RGB2BGR)    

        # skip if now prediction results
        if not bboxes:
            colors = None
        else:
            num_objs = len(bboxes.bbox)
            hsv = [(x / num_objs, 1., 1.) for x in range(num_objs)]
            colors = np.asarray([colorsys.hsv_to_rgb(*x) for x in hsv]) * 255
            
        if viz_2D:
            rgb_2d = self.visualize_2D(rgb,
                                bboxes,
                                dataset.labels,
                                score_thresh=score_thresh,
                                colors=colors)
            plots['rgb_2d'] = rgb_2d
        
        if 'mask' in bboxes.extra_fields:
            self.overlay_mask(rgb_2d, bboxes, colors)

        return plots
        
    def visualize_2D(self,
                    rgb,
                    bboxes,
                    labels,
                    score_thresh=0.25,
                    colors=None):
        if not bboxes:
            return rgb
            
        # if colors is None:
        #     hsv = [(x / len(bboxes), 1., 1.) for x in range(len(bboxes))]
        #     colors = np.asarray([colorsys.hsv_to_rgb(*x) for x in hsv]) * 255

        rgb_2d = self.draw_2D(rgb, 
                        bboxes, 
                        labels, 
                        colors=colors,
                        score_thresh=score_thresh,
                        ignored_label=[0])
        return rgb_2d

    def draw_2D(self, 
                image, 
                detections, 
                classid_map, 
                colors,
                score_thresh=0.05, 
                ignored_label=[]):
        '''
        image: (h, w, 3)
        detections: BoxList, the detections.bbox are already denormalized

        '''
        bboxes = detections.bbox.detach().cpu().numpy()
        classes = detections.extra_fields['labels'].detach().cpu().numpy()
        if 'scores' in detections.extra_fields:
            scores = detections.extra_fields['scores'].detach().cpu().numpy()
        else:
            scores = detections.extra_fields['objectness'].detach().cpu().numpy()
        # detections.extra_fields['']
        h, w, _ = image.shape
        out_box2d = np.copy(image)

        for idx in range(len(bboxes)):
            if scores[idx] < score_thresh:
                continue
            bbox = bboxes[idx]
            rgb = tuple(colors[idx])
            # Draw 2D box, banner and text
            lt = (int(bbox[0]), int(bbox[1]))
            rb = (int(bbox[2]), int(bbox[3]))
            if int(classes[idx]) in ignored_label:
                continue
            class_label = classid_map[int(classes[idx])]
            text = '{}: {:.2f}'.format(class_label, scores[idx])
            cv2.rectangle(out_box2d, lt, rb, rgb, 2)
            cv2.rectangle(out_box2d, (lt[0], lt[1] - 20), (rb[0], lt[1]), rgb, -1)
            cv2.putText(out_box2d, text, (lt[0] + 5, lt[1] - 5), 0, 0.5, (0, 0, 0), 2)

        return out_box2d

    def overlay_mask(self, image, predictions, colors=None):
        """
        Adds the instances contours for each predicted object.
        Each label has a different color.
        Arguments:
            image (np.ndarray): an image as returned by OpenCV
            predictions (BoxList): the result of the computation by the model.
                It should contain the field `mask` and `labels`.
        """
        masks = predictions.get_field("mask").numpy()
        labels = predictions.get_field("labels")
        
        if colors is None:
            colors = self.compute_colors_for_labels(labels).tolist()

        for mask, color in zip(masks, colors):
            thresh = mask[0, :, :, None]
            thresh = (thresh*255).astype(np.uint8)
            contours, hierarchy = cv2_util.findContours(
                thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
            )
            
            image = cv2.drawContours(image, contours, -1, color, 3)

        composite = image

        return composite
    def compute_colors_for_labels(self, labels):
        """
        Simple function that adds fixed colors depending on the class
        """
        colors = labels[:, None] * self.palette
        colors = (colors % 255).numpy().astype("uint8")
        return colors
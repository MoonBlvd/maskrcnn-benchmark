'''
Combine DoTA and BDD100K to an object detection dataset
'''
import os
from PIL import Image
import torch
import torchvision
from torch.utils.data import Dataset
from maskrcnn_benchmark.structures.bounding_box import BoxList
import glob
import json
# from detection.structures.segmentation_mask import SegmentationMask
# from detection.structures.keypoint import PersonKeypoints
import pdb
from tqdm import tqdm
def _count_visible_keypoints(anno):
    return sum(sum(1 for v in ann["keypoints"][2::3] if v > 0) for ann in anno)


def _has_only_empty_bbox(anno):
    return all(any(o <= 1 for o in obj["bbox"][2:]) for obj in anno)


def has_valid_annotation(anno):
    # if it's empty, there is no annotation
    if len(anno) == 0:
        return False
    # if all boxes have close to zero area, there is no annotation
    if _has_only_empty_bbox(anno):
        return False
    # keypoints task have a slight different critera for considering
    # if an annotation is valid
    if "keypoints" not in anno[0]:
        return True
    # for keypoint detection tasks, only consider valid images those
    # containing at least min_keypoints_per_image
    if _count_visible_keypoints(anno) >= min_keypoints_per_image:
        return True
    return False

class BDD100KPlusDoTA(Dataset):
    '''
    '''
    CLASSES = {
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

    def __init__(
        self, root, transforms=None, remove_images_without_annotations=False
    ):
        
        super(BDD100KPlusDoTA, self).__init__()
        self.root = root
        self.all_labels = []
        # sort indices for reproducible results        
        all_image_folders = sorted(glob.glob(os.path.join(root, 'frames/*')))
        for image_folder in tqdm(all_image_folders):
            vid = image_folder.split('/')[-1]
            all_image_paths = sorted(glob.glob(os.path.join(image_folder, '*.jpg')))
            for image_path in all_image_paths:
                anno = {
                        'image_path': image_path,
                        'objects': [],
                        'accident_id': 0,
                        'frame_id': int(image_path.split('/')[-1][:-4])}
                self.all_labels.append([vid, anno])

        self.transforms = transforms
        self.labels = list(BDD100KPlusDoTA.CLASSES.keys())
    def __getitem__(self, idx):
        '''
        Returns:
            imgs: (T, 3, H, W)
            target: annotation of the bboxes on the last frame of imgs.
        '''
        video_name, anno = self.all_labels[idx]
        
        #NOTE: the img_path is wrong, need to fix
        # load img
        img = Image.open(anno['image_path']).convert('RGB')
        
        # img = Image.open(anno['image_path']).convert('RGB')

        bboxes = []
        classes = []
        for obj in anno['objects']:
            if obj['category ID'] > 0:
                bboxes.append(torch.LongTensor(obj['bbox'])) # ltbr
                classes.append(obj['category ID'])

        if len(bboxes) > 0:
            try:
                bboxes = torch.stack(bboxes, dim=0)
            except:
                print(bboxes)
            classes = torch.LongTensor(classes)
            target = BoxList(bboxes, img.size, mode="xyxy")
            target.add_field("labels", classes)
            if anno['accident_id'] > 0:
                target.add_field("anomaly", 1.0)
            else:
                target.add_field("anomaly", 0.0)
            
            # Do not clip given we are running test only
            # target = target.clip_to_image(remove_empty=True)
            
        else:
            target = BoxList(torch.zeros(0,4), img.size, mode="xyxy") 
            target.add_field('labels', torch.zeros(0))
            target.add_field('anomaly', float(anno['accident_id'] > 0))
        
        if self.transforms is not None: 
            img, target = self.transforms(img, target)  
        
        # For test data, put 
        target.add_field('video_name', video_name)
        target.add_field('frame_id', anno['frame_id'])

        return img, target, idx

    def __len__(self):
        return len(self.all_labels)
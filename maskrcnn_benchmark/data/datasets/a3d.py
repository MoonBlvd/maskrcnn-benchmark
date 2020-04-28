'''
Combine A3D and BDD100K to an object detection dataset
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

class A3DDataset(Dataset):
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
        self, ann_file, split_file, root, transforms=None, remove_images_without_annotations=False
    ):
        self.root = root
        super(A3DDataset, self).__init__()
        # sort indices for reproducible results
        self.all_anno_files = sorted(glob.glob(ann_file))
        with open(split_file, 'r') as f:
            self.split_video_names = f.read().splitlines()
        
        self.all_labels = []
        for anno_file in self.all_anno_files:
            video_name = anno_file.split('/')[-1].split('.')[0]
            if video_name in self.split_video_names:
                annos = json.load(open(anno_file, 'r'))
                for anno in annos['labels']:
                    # NOTE: Dec 22, add function to ignore images we already processed!
                    output_dir = '/u/bryao/work/DATA/A3D_2.0/detection_with_seg/' #'/home/data/vision7/A3D_2.0/detection/'
                    save_dir = os.path.join(output_dir, video_name)
                    save_dir = os.path.join(save_dir, str(anno['frame_id']).zfill(6)+'.pth')
                    if os.path.exists(save_dir):
                        continue
                    self.all_labels.append([video_name, anno])
        self.transforms = transforms
        self.labels = list(A3DDataset.CLASSES.keys())
        # pdb.set_trace()

    def __getitem__(self, idx):
        '''
        Returns:
            imgs: (T, 3, H, W)
            target: annotation of the bboxes on the last frame of imgs.
        '''
        video_name, anno = self.all_labels[idx]
        # img_path = anno['image_path']
        #NOTE: the img_path is wrong, need to fix
        img_path = ''
        for _dir in anno['image_path'].split('/')[-4:]:
            img_path = os.path.join(img_path, _dir)
        # load img
        img = Image.open(os.path.join(self.root, img_path)).convert('RGB')

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
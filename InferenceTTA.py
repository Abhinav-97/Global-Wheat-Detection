import numpy as np  # linear algebra
import pandas as pd
from glob import glob
import os
from itertools import product
import cv2
import gc
import tqdm
from matplotlib import pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

from effdet import get_efficientdet_config, EfficientDet, DetBenchEval
from effdet.efficientdet import HeadNet
import ensemble_boxes

parser.add_argument('--test_dir', type=str, default='test', help='test directory path')
parser.add_argument('--model_path', type=str, default='', help='model_weights_path')
parser.add_argument('--image_size', type=int, default=1024, help='image Size')

args = vars(parser.parse_args())

DATA_ROOT_PATH = args["test_dir"]
IMG_SIZE = args["image_size"]


def get_valid_transforms():
    return A.Compose([
        A.Resize(height=IMG_SIZE, width=IMG_SIZE, p=1.0),
        ToTensorV2(p=1.0),
    ], p=1.0)


class DatasetRetriever(Dataset):

    def __init__(self, image_ids, transforms=None):
        super().__init__()
        self.image_ids = image_ids
        self.transforms = transforms

    def __getitem__(self, index: int):
        image_id = self.image_ids[index]
        image = cv2.imread(f'{DATA_ROOT_PATH}/{image_id}.jpg', cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0
        if self.transforms:
            sample = {'image': image}
            sample = self.transforms(**sample)
            image = sample['image']
        return image, image_id

    def __len__(self) -> int:
        return self.image_ids.shape[0]


dataset = DatasetRetriever(
    image_ids=np.array([path.split('/')[-1][:-4] for path in glob(f'{DATA_ROOT_PATH}/*.jpg')]),
    transforms=get_valid_transforms()
)


def collate_fn(batch):
    return tuple(zip(*batch))


data_loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=False,
    num_workers=2,
    drop_last=False,
    collate_fn=collate_fn
)


def load_net(checkpoint_path):
    config = get_efficientdet_config('tf_efficientdet_d5')
    net = EfficientDet(config, pretrained_backbone=False)

    config.num_classes = 1
    config.image_size = IMG_SIZE
    net.class_net = HeadNet(config, num_outputs=config.num_classes, norm_kwargs=dict(eps=.001, momentum=.01))

    checkpoint = torch.load(checkpoint_path)
    net.load_state_dict(checkpoint['model_state_dict'])

    del checkpoint
    gc.collect()

    net = DetBenchEval(net, config)
    net.eval();
    return net.cuda()


net = load_net(args["model_path"])


class BaseWheatTTA:

    image_size = IMG_SIZE

    def augment(self, image):
        raise NotImplementedError

    def batch_augment(self, images):
        raise NotImplementedError

    def deaugment_boxes(self, boxes):
        raise NotImplementedError

""" author: @shonenkov """
class TTAHorizontalFlip(BaseWheatTTA):

    def augment(self, image):
        return image.flip(1)

    def batch_augment(self, images):
        return images.flip(2)

    def deaugment_boxes(self, boxes):
        boxes[:, [1, 3]] = self.image_size - boxes[:, [3, 1]]
        return boxes


class TTAVerticalFlip(BaseWheatTTA):

    def augment(self, image):
        return image.flip(2)

    def batch_augment(self, images):
        return images.flip(3)

    def deaugment_boxes(self, boxes):
        boxes[:, [0, 2]] = self.image_size - boxes[:, [2, 0]]
        return boxes


class TTARotate90(BaseWheatTTA):

    def augment(self, image):
        return torch.rot90(image, 1, (1, 2))

    def batch_augment(self, images):
        return torch.rot90(images, 1, (2, 3))

    def deaugment_boxes(self, boxes):
        res_boxes = boxes.copy()
        res_boxes[:, [0, 2]] = self.image_size - boxes[:, [1, 3]]
        res_boxes[:, [1, 3]] = boxes[:, [2, 0]]
        return res_boxes


class TTACompose(BaseWheatTTA):

    def __init__(self, transforms):
        self.transforms = transforms

    def augment(self, image):
        for transform in self.transforms:
            image = transform.augment(image)
        return image

    def batch_augment(self, images):
        for transform in self.transforms:
            images = transform.batch_augment(images)
        return images

    def prepare_boxes(self, boxes):
        result_boxes = boxes.copy()
        result_boxes[:, 0] = np.min(boxes[:, [0, 2]], axis=1)
        result_boxes[:, 2] = np.max(boxes[:, [0, 2]], axis=1)
        result_boxes[:, 1] = np.min(boxes[:, [1, 3]], axis=1)
        result_boxes[:, 3] = np.max(boxes[:, [1, 3]], axis=1)
        return result_boxes

    def deaugment_boxes(self, boxes):
        for transform in self.transforms[::-1]:
            boxes = transform.deaugment_boxes(boxes)
        return self.prepare_boxes(boxes)


def process_det(index, det, score_threshold=0.25):
    boxes = det[index].detach().cpu().numpy()[:, :4]
    scores = det[index].detach().cpu().numpy()[:, 4]
    boxes[:, 2] = boxes[:, 2] + boxes[:, 0]
    boxes[:, 3] = boxes[:, 3] + boxes[:, 1]
    boxes = (boxes).clip(min=0, max=511).astype(int)
    indexes = np.where(scores > score_threshold)
    boxes = boxes[indexes]
    scores = scores[indexes]
    return boxes, scores


tta_transforms = []
for tta_combination in product([TTAHorizontalFlip(), None],
                               [TTAVerticalFlip(), None],
                               [TTARotate90(), None]):
    tta_transforms.append(TTACompose([tta_transform for tta_transform in tta_combination if tta_transform]))


def make_tta_predictions(images, score_threshold=0.25):
    with torch.no_grad():
        images = torch.stack(images).float().cuda()
        predictions = []
        for tta_transform in tta_transforms:
            result = []
            det = net(tta_transform.batch_augment(images.clone()), torch.tensor([1] * images.shape[0]).float().cuda())

            for i in range(images.shape[0]):
                boxes = det[i].detach().cpu().numpy()[:, :4]
                scores = det[i].detach().cpu().numpy()[:, 4]
                indexes = np.where(scores > score_threshold)[0]
                boxes = boxes[indexes]
                boxes[:, 2] = boxes[:, 2] + boxes[:, 0]
                boxes[:, 3] = boxes[:, 3] + boxes[:, 1]
                boxes = tta_transform.deaugment_boxes(boxes.copy())
                result.append({
                    'boxes': boxes,
                    'scores': scores[indexes],
                })
            predictions.append(result)
    return predictions


def run_wbf(predictions, image_index, image_size=IMG_SIZE, iou_thr=0.44, skip_box_thr=0.43, weights=None):
    boxes = [(prediction[image_index]['boxes'] / (image_size - 1)).tolist() for prediction in predictions]
    scores = [prediction[image_index]['scores'].tolist() for prediction in predictions]
    labels = [np.ones(prediction[image_index]['scores'].shape[0]).astype(int).tolist() for prediction in predictions]
    boxes, scores, labels = ensemble_boxes.ensemble_boxes_wbf.weighted_boxes_fusion(boxes, scores, labels, weights=None,
                                                                                    iou_thr=iou_thr,
                                                                                    skip_box_thr=skip_box_thr)
    boxes = boxes * (image_size - 1)
    return boxes, scores, labels


def format_prediction_string(boxes, scores):
    pred_strings = []
    for j in zip(scores, boxes):
        pred_strings.append("{0:.4f} {1} {2} {3} {4}".format(j[0], j[1][0], j[1][1], j[1][2], j[1][3]))
    return " ".join(pred_strings)


results = []

for images, image_ids in data_loader:
    predictions = make_tta_predictions(images)
    for i, image in enumerate(images):
        boxes, scores, labels = run_wbf(predictions, image_index=i)
        boxes = (boxes * 2).round().astype(np.int32).clip(min=0, max=1023)
        image_id = image_ids[i]

        boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
        boxes[:, 3] = boxes[:, 3] - boxes[:, 1]

        result = {
            'image_id': image_id,
            'PredictionString': format_prediction_string(boxes, scores)
        }
        results.append(result)

test_df = pd.DataFrame(results, columns=['image_id', 'PredictionString'])
test_df.to_csv('submission.csv', index=False)
test_df.head()

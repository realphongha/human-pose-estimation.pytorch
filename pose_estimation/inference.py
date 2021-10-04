# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run inference on images, videos, directories, streams, etc.

Usage:
    $ python path/to/detect.py --source path/to/img.jpg --weights yolov5s.pt --img 640
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
from tqdm import tqdm
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0].parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
RELATIVE_ROOT = ROOT.relative_to(Path.cwd())  # relative

sys.path.append(str(os.path.join(ROOT, "yolov5")))
from yolov5.models.experimental import attempt_load


import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as T

sys.path.insert(0, str(os.path.join(ROOT, "lib")))
# print(sys.path)
from lib.core.config import config
from lib.core.config import update_config
from lib.core.config import update_dir
from lib.core.loss import JointsMSELoss
from lib.core.function import validate
from lib.utils.utils import create_logger

import models

import argparse

import cv2
import numpy as np

from infer_utils.boxes import letterbox, scale_boxes, non_max_suppression, xyxy2xywh
from infer_utils.decode import get_final_preds, get_simdr_final_preds
from infer_utils.utils import setup_cudnn, get_affine_transform, draw_keypoints
from infer_utils.utils import VideoReader, VideoWriter, WebcamStream, FPS


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pose_model', type=str, default=None, help='path to pretrained pose model')
    parser.add_argument('--pose_cfg', type=str, default=None, help='path to pose model configuration file')
    parser.add_argument('--det_model', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    return opt


def reset_config(config, args):
    # if args.gpus:
    #     config.GPUS = args.gpus
    # if args.workers:
    #     config.WORKERS = args.workers
    # if args.use_detect_bbox:
    #     config.TEST.USE_GT_BBOX = not args.use_detect_bbox
    # if args.flip_test:
    #     config.TEST.FLIP_TEST = args.flip_test
    # if args.post_process:
    #     config.TEST.POST_PROCESS = args.post_process
    # if args.shift_heatmap:
    #     config.TEST.SHIFT_HEATMAP = args.shift_heatmap
    if args.pose_model:
        config.TEST.MODEL_FILE = args.pose_model
    # if args.coco_bbox_file:
    #     config.TEST.COCO_BBOX_FILE = args.coco_bbox_file
    
    
class Pose:

    SKELETONS = {"coco":[
            [16,14], [14,12], [17,15], [15,13], [12,13], [6,12], [7,13], 
            [6,7], [6,8], [7,9], [8,10], [9,11], [2,3], [1,2], [1,3], [2,4], 
            [3,5], [4,6], [5,7]
        ],
        "mpii": [[9, 10], [12, 13], [12, 11], [3, 2], [2, 1], [14, 15], 
                 [15, 16], [4, 5], [5, 6], [9, 8], [8, 7], [7, 3], [7, 4], 
                 [9, 13], [9, 14]
        ]}

    def __init__(self, 
        det_model,
        pose_model,
        device,
        model_name,
        dataset_name,
        img_size=640,
        conf_thres=0.25,
        iou_thres=0.45, 
    ) -> None:
        self.img_size = img_size
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.det_model = det_model
        self.pose_model = pose_model
        self.device = device
        self.model_name = model_name

        self.patch_size = (192, 256)

        self.pose_transform = T.Compose([
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

        try:
            self.skeleton = Pose.SKELETONS[dataset_name.lower()]
        except KeyError:
            self.skeleton = None
        

    def preprocess(self, image):   
        img = letterbox(image, new_shape=self.img_size)
        img = np.ascontiguousarray(img.transpose((2, 0, 1)))
        img = torch.from_numpy(img).to(self.device)
        img = img.float() / 255.0
        img = img[None]
        return img

    def box_to_center_scale(self, boxes, pixel_std=200):
        boxes = xyxy2xywh(boxes)
        r = self.img_size[0] / self.img_size[1]
        mask = boxes[:, 2] > boxes[:, 3] * r
        boxes[mask, 3] = boxes[mask, 2] / r
        boxes[~mask, 2] = boxes[~mask, 3] * r
        boxes[:, 2:] /= pixel_std 
        boxes[:, 2:] *= 1.25
        return boxes

    def predict_poses(self, boxes, img):
        image_patches = []
        for cx, cy, w, h in boxes:
            trans = get_affine_transform(np.array([cx, cy]), np.array([w, h]), self.img_size)
            img_patch = cv2.warpAffine(
                img, 
                trans, 
                (int(self.img_size[0]), int(self.img_size[1])), 
                flags=cv2.INTER_LINEAR)
            img_patch = self.pose_transform(img_patch)
            image_patches.append(img_patch)

        image_patches = torch.stack(image_patches).to(self.device)
        return self.pose_model(image_patches)

    def postprocess(self, pred, img1, img0):
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes=0)

        for det in pred:

            if len(det):
                boxes = scale_boxes(det[:, :4], img0.shape[:2], img1.shape[-2:]).cpu()
                for box in boxes.numpy():
                  x1, y1, x2, y2 = box
                  x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                  img0 = cv2.rectangle(img0, (x1, y1), (x2, y2), (255, 0, 0), 1)
                boxes = self.box_to_center_scale(boxes)
                outputs = self.predict_poses(boxes, img0)

                if 'simdr' in self.model_name:
                    coords = get_simdr_final_preds(*outputs, boxes, self.img_size)
                else:
                    coords = get_final_preds(outputs, boxes)
                    
                draw_keypoints(img0, coords, self.skeleton, 1)

    @torch.no_grad()
    def predict(self, image):
        img = self.preprocess(image)
        pred = self.det_model(img)[0]  
        self.postprocess(pred, img, image)
        return image


def main(opt):
    det_model = attempt_load(opt.det_model)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    det_model = det_model.to(device)
    det_model.eval()
    
    # update config
    update_config(opt.pose_cfg)
    reset_config(config, opt)
    from lib.models import MODELS    
    pose_model = MODELS[config.MODEL.NAME](config, is_train=False)
    state_dict = torch.load(opt.pose_model)
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if "module." in k:
            name = k[7:]  # remove "module"
        else:
            name = k
        new_state_dict[name] = v
    state_dict = new_state_dict
    pose_model.load_state_dict(state_dict)
    pose_model.to(device)
    pose_model.eval()
    
    pose = Pose(det_model, pose_model, device, config.MODEL.NAME, 
                config.DATASET.DATASET, config.MODEL.IMAGE_SIZE)
    source = Path(opt.source)
    if source.is_file() and source.suffix in ['.jpg', '.jpeg', '.png']:
        image = cv2.imread(str(source), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        output = pose.predict(image)
        cv2.imwrite(f"{str(source).rsplit('.', maxsplit=1)[0]}_out.jpg", cv2.cvtColor(output, cv2.COLOR_RGB2BGR))

    elif source.is_dir():
        files = source.glob("*.jpg")
        for file in files:
            image = cv2.imread(str(file), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            output = pose.predict(image)
            cv2.imwrite(f"{str(file).rsplit('.', maxsplit=1)[0]}_out.jpg", cv2.cvtColor(output, cv2.COLOR_RGB2BGR))

    elif source.is_file() and source.suffix in ['.mp4', '.avi', '.mkv']:
        video_out = f"{opt.source.rsplit('.', maxsplit=1)[0]}_out.mp4"
        video_reader = cv2.VideoCapture(opt.source)

        nb_frames = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_h = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_w = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
        fps = int(video_reader.get(cv2.CAP_PROP_FPS))
        
        video_writer = cv2.VideoWriter(video_out,
                                cv2.VideoWriter_fourcc(*'MPEG'), 
                                fps, 
                                (frame_w, frame_h))

        for i in tqdm(range(nb_frames)):
            _, frame = video_reader.read()
            if frame is None:
                continue
            frame = pose.predict(frame)
            video_writer.write(np.uint8(frame))

        video_reader.release()
        video_writer.release()
        
        # reader = VideoReader(opt.source)
        # writer = VideoWriter(f"{opt.source.rsplit('.', maxsplit=1)[0]}_out.mp4", reader.fps)
        # print("bbb")
        # fps = FPS(len(reader.frames))

        # for frame in tqdm(reader):
        #     fps.start()
        #     output = pose.predict(frame.numpy())
        #     fps.stop(False)
        #     writer.update(output)
        
        # print(f"FPS: {fps.fps}")
        # writer.write()

    else:
        webcam = WebcamStream()
        fps = FPS()

        for frame in webcam:
            fps.start()
            output = pose.predict(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            fps.stop()
            cv2.imshow('frame', cv2.cvtColor(output, cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
import numpy as np

# インストールディレクトリの親の親がワークスペースと仮定
workspace = Path("/home/go/slope_ws/src")
print('workspace:', workspace)
# ワークスペースの直下にディレクトリyolov8があることを仮定
ROOT = workspace / 'YOLODataset'
print('ROOT:', ROOT)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

# ファイルの先頭でimportしていないのでFlake8に怒られる（E402）
# yolov5で用意されたモジュールをyolov8でも使えるか互換性を見て行っている。
from ultralytics.engine.model import Model
from ultralytics.utils.ops import (scale_coords, non_max_suppression,)
#check_imgszがcheck_img_sizeの代用
from ultralytics.utils.checks import (check_imgsz, check_imshow, check_requirements, print_args)
from ultralytics.utils.plotting import Annotator, colors
from ultralytics.utils.torch_utils import select_device
from ultralytics.data.augment import LetterBox


class Result:
    def __init__(self, xyxy=(0.0, 0.0, 0.0, 0.0), name='', conf=0.0):
        self.u1 = float(xyxy[0])
        self.v1 = float(xyxy[1])
        self.u2 = float(xyxy[2])
        self.v2 = float(xyxy[3])
        self.name = name
        self.conf = float(conf)


class Detectorv8:

    def __init__(
        self,
        weights=ROOT / 'best.pt',  # model.pt path(s)
        #data=ROOT / 'dataset.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.8,  # confidence threshold
        iou_thres=0.65,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
    ):
        check_requirements(exclude=('tensorboard', 'thop'))
        # Load model
        self.device = select_device(device)
        self.model = Model(weights)#data引数削除
        self.stride = self.model.stride
        self.names = self.model.names
        self.pt = self.model.pt
        self.jit = self.model.jit
        self.onnx = self.model.onnx
        self.engine = self.model.engine
        self.imgsz = check_imgsz(imgsz, s=self.stride)  # check image size
        self.view_img = view_img
        self.augment = augment
        self.visualize = visualize
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.classes = classes
        self.agnostic_nms = agnostic_nms
        self.max_det = max_det
        self.line_thickness = line_thickness
        self.hide_labels = hide_labels
        self.hide_conf = hide_conf
        self.half = half

        # Half
        # FP16 supported on limited backends with CUDA
        self.half &= ((self.pt or self.jit or self.onnx or self.engine)
                      and self.device.type != 'cpu')
        if self.pt or self.jit:
            self.model.model.half() if self.half else self.model.model.float()

        # Dataloader
        self.view_img = True  # check_imshow()
        # set True to speed up constant image size inference
        cudnn.benchmark = True
        self.model.warmup(imgsz=(1, 3, *imgsz))  # warmup

    @torch.no_grad()
    def detect(self, img0):
        img = LetterBox(img0, self.imgsz, stride=self.stride)[0]

        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255  # 0 - 255 to 0.0 - 1.0
        img = img[None]  # expand for batch dim

        # Inference
        pred = self.model(img, augment=self.augment, visualize=self.visualize)

        # NMS
        pred = non_max_suppression(
            pred, self.conf_thres, self.iou_thres, self.classes,
            self.agnostic_nms, max_det=self.max_det)

        det = pred[0]
        s = '%gx%g ' % img.shape[2:]  # print string
        torch.tensor(img0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        annotator = Annotator(
            img0, line_width=self.line_thickness, example=str(self.names))

        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(
                img.shape[2:], det[:, :4], img0.shape).round()

            # Print results
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class
                # add to string
                s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "
            # Write results
            result = []
            for *xyxy, conf, cls in reversed(det):
                result.append(Result(xyxy, self.names[int(cls)], conf))
                if self.view_img:  # Add bbox to image
                    c = int(cls)  # integer class
                    label = None if self.hide_labels else (
                        self.names[c] if self.hide_conf else (
                            f'{self.names[c]} {conf:.2f}'))
                    annotator.box_label(xyxy, label, color=colors(c, True))

            return img0, result
        else:
            return img0, []

def parse_opt(args):
    sys.argv = args
    parser = argparse.ArgumentParser()
    #ArgumentParserのインスタンスを作成し、引数の解析を行う準備をします。
    parser.add_argument(
        '--weights', nargs='+', type=str, default=ROOT / 'best.pt',
        help='model path(s)')
    # parser.add_argument(
    #     '--data', type=str, default=ROOT / 'dataset.yaml',
    #     help='(optional) dataset.yaml path')
    parser.add_argument(
        '--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640],
        help='inference size h,w')
    parser.add_argument(
        '--conf-thres', type=float, default=0.8, help='confidence threshold')
    parser.add_argument(
        '--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument(
        '--max-det', type=int, default=1000,
        help='maximum detections per image')
    parser.add_argument(
        '--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument(
        '--view-img', action='store_true', help='show results')
    parser.add_argument(
        '--classes', nargs='+', type=int,
        help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument(
        '--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument(
        '--augment', action='store_true', help='augmented inference')
    parser.add_argument(
        '--visualize', action='store_true', help='visualize features')
    parser.add_argument(
        '--line-thickness', default=3, type=int,
        help='bounding box thickness (pixels)')
    parser.add_argument(
        '--hide-labels', default=False, action='store_true',
        help='hide labels')
    parser.add_argument(
        '--hide-conf', default=False, action='store_true',
        help='hide confidences')
    parser.add_argument(
        '--half', action='store_true',
        help='use FP16 half-precision inference')
    # parser.add_argument(
    #     '--dnn', action='store_true',
    #     help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt
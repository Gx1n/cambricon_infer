"""
 * @Author: GxIn
 * @Data: 10/17/25 1:39?PM
 * @Description: 测试YOLOv10模型在寒武纪设备上的推理性能，包括预处理、推理、后处理各阶段耗时
"""
import time
import cv2
import os
import torch
import torch_mlu
import numpy as np
from typing import Union, List

from ultralytics.data.augment import LetterBox
from ultralytics.models.yolo.detect import DetectionPredictor
from ultralytics.nn.tasks import yaml_model_load, DetectionModel
from ultralytics.utils import ops

# 类别颜色配置
class_colors = {
    0: (0, 0, 255),    # fire - 红色
    1: (255, 0, 0),    # smoke - 蓝色
}

# 类别名称
names = ["fire", "smoke"]

class YOLOv10DetectionPredictor(DetectionPredictor):
    def postprocess(self, preds, img, orig_img):
        if isinstance(preds, dict):
            preds = preds["one2one"]
        if isinstance(preds, (list, tuple)):
            preds = preds[0]
        if preds.shape[-1] == 6:
            pass
        else:
            preds = preds.transpose(-1, -2)
            bboxes, scores, labels = ops.v10postprocess(preds, self.args.max_det, preds.shape[-1] - 4)
            bboxes = ops.xywh2xyxy(bboxes)
            preds = torch.cat([bboxes, scores.unsqueeze(-1), labels.unsqueeze(-1)], dim=-1)
        mask = preds[..., 4] > self.args.conf
        if self.args.classes is not None:
            mask = mask & (preds[..., 5:6] == torch.tensor(self.args.classes, device=preds.device).unsqueeze(0)).any(2)
        preds = [p[mask[idx]] for idx, p in enumerate(preds)]
        return preds

# def postprocess(preds, img, orig_img):
#     if isinstance(preds, dict):
#         preds = preds["one2one"]
#     if isinstance(preds, (list, tuple)):
#         preds = preds[0]
#     if preds.shape[-1] == 6:
#         pass
#     else:
#         preds = preds.transpose(-1, -2)
#         bboxes, scores, labels = ops.v10postprocess(preds, self.args.max_det, preds.shape[-1] - 4)
#         bboxes = ops.xywh2xyxy(bboxes)
#         preds = torch.cat([bboxes, scores.unsqueeze(-1), labels.unsqueeze(-1)], dim=-1)
#     mask = preds[..., 4] > self.args.conf
#     if self.args.classes is not None:
#         mask = mask & (preds[..., 5:6] == torch.tensor(self.args.classes, device=preds.device).unsqueeze(0)).any(2)
#     preds = [p[mask[idx]] for idx, p in enumerate(preds)]
#     return preds

postprocess_yolov10 = YOLOv10DetectionPredictor()

def pre_transform(im: List[np.ndarray]) -> List[np.ndarray]:
    """
    对输入图像进行预变换处理
    """
    letterbox = LetterBox([640, 640], auto=True, stride=32)
    return [letterbox(image=im)]

def preprocess(im: Union[torch.Tensor, List[np.ndarray]]) -> torch.Tensor:
    """
    图像预处理函数
    """
    not_tensor = not isinstance(im, torch.Tensor)
    if not_tensor:
        im = np.stack(pre_transform(im))
        if im.shape[-1] == 3:
            im = im[..., ::-1]  # BGR to RGB
        im = im.transpose((0, 3, 1, 2))  # BHWC to BCHW, (n, 3, h, w)
        im = np.ascontiguousarray(im)  # contiguous
        im = torch.from_numpy(im).float()

    im = im.to('cuda')
    if not_tensor:
        im /= 255  # 0 - 255 to 0.0 - 1.0
    return im

def construct_result(pred, img, orig_img):
    """
    构建最终结果，调整边界框坐标适应原始图像尺寸
    """
    pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
    return pred

def draw_detections(img, preds, names, class_colors):
    """
    在图像上绘制检测结果
    """
    for obj in preds:
        left, top, right, bottom = int(obj[0]), int(obj[1]), int(obj[2]), int(obj[3])
        confidence = obj[4]
        label = int(obj[5])
        color = class_colors.get(label, (0, 255, 255))
        cv2.rectangle(img, (left, top), (right, bottom), color=color, thickness=2, lineType=cv2.LINE_AA)
        caption = f"{names[label]} {confidence:.2f}"
        w, h = cv2.getTextSize(caption, 0, 1, 1)[0]
        cv2.rectangle(img, (left - 3, top - 33), (left + w + 10, top), color, -1)
        cv2.putText(img, caption, (left, top - 5), 0, 1, (0, 0, 0), 1, 16)
    return img

def load_model(config_path, weight_path):
    """
    加载模型
    """
    model = DetectionModel(cfg=yaml_model_load(config_path))
    ckpt = torch.load(weight_path)
    model.load(ckpt)
    model.eval()
    model.to('cuda')
    return model

def run_inference(model, frame):
    """
    运行完整推理流程并测量各阶段耗时
    """
    # 读取图像
    # image = cv2.imread(img_path, cv2.IMREAD_COLOR)
    ori_image = frame.copy()
    
    # 预处理阶段计时
    preprocess_start = time.time()
    processed_image = preprocess(frame)
    preprocess_time = time.time() - preprocess_start
    
    # 推理阶段计时
    infer_start = time.time()
    preds = model(processed_image)
    infer_time = time.time() - infer_start
    
    # 后处理阶段计时
    postprocess_start = time.time()
    # postprocess_yolov10 = YOLOv10DetectionPredictor()
    preds = postprocess_yolov10.postprocess(preds, torch.tensor(processed_image), processed_image)
    preds = construct_result(preds[0], processed_image, ori_image)
    postprocess_time = time.time() - postprocess_start
    
    # 打印各阶段耗时
    #print(f"Preprocess time: {preprocess_time:.4f}s")
    #print(f"Inference time: {infer_time:.4f}s")
    #print(f"Postprocess time: {postprocess_time:.4f}s")
    print(f"Total time: {preprocess_time + infer_time + postprocess_time:.4f}s")
    
    return preds, ori_image

def main():
    # 模型路径配置（使用相对路径）
    config_path = "./cfg/yolov10m.yaml"
    weight_path = ""
    output_dir = "./detect_res"
    # output_path = os.path.join(output_dir, "result.jpg")
    
    # 加载模型
    print("Loading model...")
    model = load_model(config_path, weight_path)
    print("Model loaded successfully.")

    cap = cv2.VideoCapture("data/test.mp4")
    c = 0
    while True:
        ret, frame = cap.read()
        if ret:
            # 运行推理
            print("Running inference...")
            preds, img0 = run_inference(model, frame)

            # 绘制检测结果
            img0 = draw_detections(img0, preds, names, class_colors)

            # 保存结果
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            cv2.imwrite(os.path.join(output_dir, f"{c}.png"), img0)
            c += 1
    # print(f"Result saved to {output_path}")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
@Filename:    deeplab_pytorch.py
@Author:      dulanj
@Time:        12/01/2022 18:31
"""
from datetime import datetime

import PIL
import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

pascal_voc_classes = {
        "0": "background",
        "1": "aeroplane",
        "2": "bicycle",
        "3": "bird",
        "4": "boat",
        "5": "bottle",
        "6": "bus",
        "7": "car",
        "8": "cat",
        "9": "chair",
        "10": "cow",
        "11": "diningtable",
        "12": "dog",
        "13": "horse",
        "14": "motorbike",
        "15": "person",
        "16": "pottedplant",
        "17": "sheep",
        "18": "sofa",
        "19": "train",
        "20": "tvmonitor"
}

def _convert_mask_to_polygon(mask):
    mask = np.array(mask, dtype=np.uint8)
    cv2.normalize(mask, mask, 0, 255, cv2.NORM_MINMAX)
    if int(cv2.__version__.split('.')[0]) > 3:
        contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)[0]
    else:
        contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)[1]

    contours = max(contours, key=lambda arr: arr.size)
    if contours.shape.count(1):
        contours = np.squeeze(contours)
    if contours.size < 3 * 2:
        raise Exception('Less then three point have been detected. Can not build a polygon.')

    polygon = []
    for point in contours:
        polygon.append([int(point[0]), int(point[1])])

    return polygon


class DeepLabv3Pytorch():
    def __init__(self):
        model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50', pretrained=True)
        # or any of these variants
        # model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet101', pretrained=True)
        # model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_mobilenet_v3_large', pretrained=True)
        model.eval()
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model = model.to(self.device)

    def _inference(self, input_image: PIL.Image):
        input_image = input_image.convert("RGB")
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        input_tensor = preprocess(input_image)
        input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model

        # move the input and model to GPU for speed if available
        input_batch = input_batch.to(self.device)

        with torch.no_grad():
            output = self.model(input_batch)['out'][0]
        output_predictions = output.argmax(0)
        return output_predictions

    def predict(self, input_image: PIL.Image):
        tic = datetime.utcnow()
        output_predictions = self._inference(input_image).to('cpu').numpy()
        results = []
        for i in range(1, 21):
            mask = np.array((output_predictions == i) * 255, dtype=np.uint8)
            if mask.sum() == 0:
                continue
            results.append({
                "confidence": "NA",
                "label": pascal_voc_classes[str(i)],
                "points": _convert_mask_to_polygon(mask),
                "type": "polygon"
            })
        toc = datetime.utcnow()
        prediction = {
            'results': results,
            'start': tic.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            'end': toc.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            'pred_time_s': round((toc - tic).total_seconds(), 3)
        }
        return prediction


if __name__ == '__main__':
    input_image = Image.open('/home/dulanj/Pictures/test/children-playing-outdoors.jpg')
    dl_model = DeepLabv3Pytorch()
    # output_predictions = dl_model._inference(input_image)
    # for i in range(20):
    #     mask = np.array((output_predictions == i) * 255, dtype=np.uint8)
    #     cv2.imshow('mask', mask)
    #     cv2.waitKey(0)
    # print(output_predictions)

    print(dl_model.predict(input_image))

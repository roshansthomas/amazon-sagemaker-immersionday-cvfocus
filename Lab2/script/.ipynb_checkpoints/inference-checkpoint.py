import os
import numpy as np
import cv2
import torchvision.models.segmentation
import torch

import json
import logging
import sys

#import sagemaker_containers
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torch.utils.data.distributed
from torchvision import datasets, transforms

from torchvision.models.segmentation.deeplabv3 import DeepLabHead

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

# Converts the incoming request image payload into a tensor
def input_fn(request_body, request_content_type):
    import io

    import torch
    import torchvision.transforms as transforms
    from PIL import Image

    print("input function")
    f = io.BytesIO(request_body)
    
    print("opening image")
    input_image = Image.open(f).convert("RGB")
    preprocess = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    
    print("transform input")
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)
    
    print("return input")
    return input_batch

#  Converts the prediction output into the response payload.
def output_fn(prediction, content_type):
    from sagemaker_inference import encoder
    
    print("content type {}".format(content_type))
    
    print("output prediction")
    print(type(prediction))
#     print("prediction shape: {}".format(prediction.shape))
    
#     print("resize")
#     prediction = transforms.Resize((650,650))(prediction['out'])
    
#     np_prediction = torch.argmax(prediction[0], 1).cpu().detach().numpy()
#     np_prediction = prediction[0].data.cpu().numpy()

#     np_prediction = prediction.cpu().detach().numpy()
#     np_prediction = torch.argmax(prediction.squeeze(), dim=1).detach().cpu().numpy()
#     np_prediction = prediction['out'].argmax(0).cpu().detach().numpy()

    prediction = torch.sigmoid(prediction['out'])[0]
    np_prediction = prediction.cpu().numpy()[0]

#     np_prediction = prediction['out'].data.cpu().numpy()
    print("np_prediction shape: {}".format(np_prediction.shape))
    print("np_prediction type: {}".format(np_prediction.dtype))
    
    return encoder.encode(np_prediction, content_type)


# Loads the model
def model_fn(model_dir):
    num_classes = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Net = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=True)  # Load net
#     Net = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=True)  # Load net
    
    Net.classifier = DeepLabHead(2048, num_classes)
#     Net.load_state_dict(torch.load(modelPath)) # Load trained model

    model = torch.nn.DataParallel(Net)
      
    with open(os.path.join(model_dir, "model.pth"), "rb") as f:
        model.load_state_dict(torch.load(f))
        
    model.eval()
    return model.to(device)
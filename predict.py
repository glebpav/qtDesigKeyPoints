import torch
import config
import cv2
import os.path
import numpy as np
import matplotlib.pyplot as plt
from model import FaceKeypointResNet50
import time

from printing_utils import *


def predict(picture_path, model_path="model_dataset9_80.pth", show_points_names=0, show_connections=0):
    predicted_points = []
    model = FaceKeypointResNet50(pretrained=True, requires_grad=False).to(config.DEVICE)
    # load the model checkpoint
    # TODO PATH TO MODEL
    checkpoint = torch.load(model_path, map_location='cpu')
    # load model weights state_dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    cap = cv2.VideoCapture(picture_path)

    # capture each frame of the video
    ret, frame = cap.read()

    with torch.no_grad():
        image = frame
        h, w = image.shape[:2]
        image = cv2.resize(image, (224, 224))
        plt.show()

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image / 255.0
        image = np.transpose(image, (2, 0, 1))
        image = torch.tensor(image, dtype=torch.float)
        image = image.unsqueeze(0).to(config.DEVICE)
        outputs = model(image)

        for i in outputs:
            x, y, confidence = 0, 0, 0
            for idx, j in enumerate(i):
                if idx % 3 == 0:
                    confidence = float(str(j)[7:len(str(j)) - 1])
                if idx % 3 == 1:
                    x = float(str(j)[7:len(str(j)) - 1])
                if idx % 3 == 2:
                    y = float(str(j)[7:len(str(j)) - 1])
                    predicted_points.append(Animals_points(x, y, confidence, animals_points_classes[int(idx / 3)]))

    outputs = outputs.cpu().detach().numpy()

    print(show_points_names, show_connections)

    # predicted
    frame = add_points(frame, predicted_points, w, h, (255, 0, 0), print_txt=show_points_names)
    if show_connections == 2:
        frame = add_connections(frame, predicted_points, w, h)

    plt.imshow(frame)
    plt.show()

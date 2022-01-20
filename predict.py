import torch
import config
import cv2
import os.path
import numpy as np
import matplotlib.pyplot as plt
from model import FaceKeypointResNet50
import time

from printing_utils import *


def predict_frame(picture_path, model_path="model_dataset9_80.pth", show_points_names=0, show_connections=0):
    predicted_points = []
    model = FaceKeypointResNet50(pretrained=True, requires_grad=False).to(config.DEVICE)
    # load the model checkpoint
    # TODO PATH TO MODEL
    checkpoint = torch.load(model_path, map_location='cpu')
    # print("hello")
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

    return frame


def video_predict(video_path, model_path="model_dataset9_80.pth", show_points_names=0, show_connections=0):

    frame_list = []

    model = FaceKeypointResNet50(pretrained=True, requires_grad=False).to(config.DEVICE)
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    cap = cv2.VideoCapture(video_path)

    ret, frame_in_video = cap.read()
    while ret:
        # cv2.imshow("output video", frame_in_video)

        predicted_points = []
        frame = frame_in_video

        height, width = frame.shape[:2]
        max_height = 1000
        max_width = 1000

        if max_height < height or max_width < width:
            # get scaling factor
            scaling_factor = max_height / float(height)
            if max_width / float(width) < scaling_factor:
                scaling_factor = max_width / float(width)
            # resize image
            frame = cv2.resize(frame, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)

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

        # cv2.imshow("output video", frame)
        frame_list.append(frame)

        ret, frame_in_video = cap.read()

        if not ret:
            break

    flag_out = False
    while True:

        if flag_out:
            break

        for frame in frame_list:
            cv2.imshow("output video", frame)

            keyCode = cv2.waitKey(1)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                flag_out = True
                break
    cap.release()
    cv2.destroyAllWindows()


def predict(picture_path, model_path="model_dataset9_80.pth", show_points_names=0, show_connections=0):
    predicted_points = []
    model = FaceKeypointResNet50(pretrained=True, requires_grad=False).to(config.DEVICE)
    # load the model checkpoint
    # TODO PATH TO MODEL
    checkpoint = torch.load(model_path, map_location='cpu')
    # print("hello")
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

import torch
import config
import numpy as np
import matplotlib.pyplot as plt
from model import *

from printing_utils import *


def load_model_and_weights(model_type):
    if model_type == "The Fastest":
        model = mobilenet_v3_large(requires_grad=False, count=90)
        checkpoint = torch.load("weights/mobilenet_v3_large.pth", map_location='cpu')

    elif model_type == "The most accurate":
        model = xception(requires_grad=False, count=90)
        checkpoint = torch.load("weights/xception.pth", map_location='cpu')

    else:
        print("selected The Balanced One/Default model")
        model = resnet50(requires_grad=False, count=90)

        from pathlib import Path
        print("selected The Balanced One/Default model 111")
        my_file = Path("weights/resnet50.pth")
        if my_file.is_file():
            print("file exits")
        else:
            print("file doesn't exits")

        checkpoint = torch.load("weights/resnet50.pth", map_location='cpu')

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model


def scale_img(frame):
    max_height = 1000
    max_width = 1000

    height, width = frame.shape[:2]

    if max_height < height or max_width < width:
        # get scaling factor
        scaling_factor = max_height / float(height)
        if max_width / float(width) < scaling_factor:
            scaling_factor = max_width / float(width)
        # resize image
        frame = cv2.resize(frame, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
        return frame


def read_video(path):
    frame_list = []
    cap = cv2.VideoCapture(path)

    print("not bug")
    # print(int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)))

    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    ret, frame_in_video = cap.read()
    while ret:
        frame = scale_img(frame_in_video)
        frame_list.append(frame)
        ret, frame_in_video = cap.read()
        if not ret:
            break

    cap.release()
    return length, fps, frame_list


def frame_skipper_factor(fps):
    avg_frame_processing = 0.2
    return round(avg_frame_processing / (1 / fps))


def predict_frame(model, frame):
    with torch.no_grad():
        print("predict_frame_start")
        image = frame
        image = cv2.resize(image, (224, 224))
        plt.show()

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image / 255.0
        image = np.transpose(image, (2, 0, 1))
        image = torch.tensor(image, dtype=torch.float)
        image = image.unsqueeze(0).to(config.DEVICE)
        outputs = model(image)

        predicted_points = []
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
        print("predict_frame_end")
        return predicted_points


def get_points_upscale(start_points, end_points, upscale_factor):
    print("point upscale start")
    output_points = []
    for idx in range(len(start_points)):
        delta_x = end_points[idx].x - start_points[idx].x
        delta_y = end_points[idx].y - start_points[idx].y
        if idx == 0:
            print("starts", start_points[idx].x)
            print("deltas", delta_x, delta_y)
        x = start_points[idx].x + upscale_factor * delta_x
        y = start_points[idx].y + upscale_factor * delta_y
        output_points.append(Animals_points(x, y, start_points[idx].confidence, start_points[idx].name_of_point))
    return output_points


def video_predict(video_path, model_type, show_points_names=0, show_connections=0):
    model = load_model_and_weights(model_type)
    video_length, fps, frame_list_input = read_video(video_path)
    frame_skipper = frame_skipper_factor(fps)
    frame_list_output = []
    points_list_output = []

    frame_counter = 0
    while frame_counter <= video_length:
        start_points = predict_frame(model, frame_list_input[frame_counter])
        points_list_output.append(start_points)
        if frame_counter + frame_skipper < video_length:
            end_points = predict_frame(model, frame_list_input[frame_counter + frame_skipper])
            for idx in range(frame_counter + 1, frame_counter + frame_skipper):
                print("here", idx, float(idx - frame_counter) / frame_skipper)
                points_list_output.append(
                    get_points_upscale(start_points, end_points, float(idx - frame_counter) / frame_skipper))
            points_list_output.append(end_points)
        frame_counter += frame_skipper

    for idx, frame in enumerate(frame_list_input):

        if idx >= len(points_list_output) - frame_skipper:
            break

        h, w = frame.shape[:2]
        predicted_points = points_list_output[idx]

        frame = add_points(frame, predicted_points, w, h, (255, 0, 0), print_txt=show_points_names)
        if show_connections == 2:
            frame = add_connections(frame, predicted_points, w, h)

        frame_list_output.append(frame)

    while True:
        for frame in frame_list_output:
            cv2.imshow("Output Video", frame)
            # cv2.delay(20)
            if cv2.waitKey(3) & 0xFF == ord('q'):
                break
        else:
            continue
        break

    cv2.destroyAllWindows()


def predict(picture_path, model_type, show_points_names=0, show_connections=0):
    predicted_points = []

    model = load_model_and_weights(model_type)

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

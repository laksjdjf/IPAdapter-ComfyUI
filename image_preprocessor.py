import torch
import torch.nn.functional as F
import os
import cv2
import subprocess

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
DETECTOR_FILE = "lbpcascade_animeface.xml"

if not os.path.exists(os.path.join(CURRENT_DIR, DETECTOR_FILE)):
    print("Downloading anime face detector...")
    try:
        subprocess.run(["wget", "https://raw.githubusercontent.com/nagadomi/lbpcascade_animeface/master/lbpcascade_animeface.xml", "-P", CURRENT_DIR])
    except:
        print("Failed to download lbpcascade_animeface.xml so please download it yourself.")

def image_to_numpy(image):
    image = image.squeeze(0) * 255
    return image.numpy().astype("uint8")

def numpy_to_image(image):
    image = torch.tensor(image).float() / 255
    return image.unsqueeze(0)

def pad_to_square(tensor):
    tensor = tensor.squeeze(0).permute(2, 0, 1)
    _, h, w = tensor.shape

    target_length = max(h, w)

    pad_l = (target_length - w) // 2
    pad_r = (target_length - w) - pad_l
    
    pad_t = (target_length - h) // 2
    pad_b = (target_length - h) - pad_t

    padded_tensor = F.pad(tensor, (pad_l, pad_r, pad_t, pad_b), mode="constant", value=0)

    return padded_tensor.permute(1, 2, 0).unsqueeze(0)

def face_crop(image):
    image = image_to_numpy(image)
    face_cascade = cv2.CascadeClassifier(os.path.join(CURRENT_DIR, DETECTOR_FILE))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    w, h = image.shape[1], image.shape[0]

    target_length = min(w, h)
    fx, fy, fw, fh = (0, 0, w, h) if len(faces) == 0 else faces[0]

    dx = target_length - fw // 2
    dy = target_length - fh // 2

    target_x = 0 if w < h else max(0, fx - dx)
    target_y = 0 if w > h else max(0, fy - dy)
    
    image = image[target_y:target_y+target_length, target_x:target_x+target_length]
    image = numpy_to_image(image)

    return image

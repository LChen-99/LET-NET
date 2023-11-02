import onnxruntime as ort
import numpy as np
from PIL import Image
import cv2


import time


# 加载 ONNX 模型
onnx_model_path = "/home/caia/lc_ws/LET/LET-NET/superpoint_240_320.onnx"
providers = ['CPUExecutionProvider']
session = ort.InferenceSession(onnx_model_path, providers=providers)

input_name = session.get_inputs()[0].name
output_name1 = session.get_outputs()[0].name

target_shape = (320, 240)
image = cv2.imread('/home/caia/lc_ws/LET/imagenet-sample-images/n01440764_tench.JPEG', cv2.IMREAD_GRAYSCALE)
image = cv2.resize(image, (320, 240))
image = np.array(image)
input =  np.reshape(image.astype(np.float32), (1, 1, 240, 320))
output_data = session.run([output_name1], {input_name: input})

   

# 输出结果
# print("Output shape:", output_data[0].shape)
# print("Output data:", output_data[0])
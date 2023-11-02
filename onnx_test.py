import onnxruntime as ort
import numpy as np
from PIL import Image
import cv2


import time


# 加载 ONNX 模型
onnx_model_path = "/home/caia/lc_ws/LET/LET-NET/superpoint_240_320.onnx"
# onnx_model_path = "./models/superpoint_detector.onnx"
# providers = ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
providers = ['CPUExecutionProvider']
session = ort.InferenceSession(onnx_model_path, providers=providers)
# 查看输入和输出节点的名称
print("input size = :", len(session.get_inputs()))
print("output size = :", len(session.get_outputs()))
input_name = session.get_inputs()[0].name
output_name1 = session.get_outputs()[0].name

print(input_name)
print(output_name1)

# 定义图像文件路径
# image_file_path = "/home/xreal/xreal/HF/hfnet/doc/demo/db1.jpg"
# 设置目标图像大小，与模型输入的大小相匹配
target_shape = (320, 240)
# 预处理图像
times = 100
total_time = 0
resize_time = 0
for i in range(times):
    image = np.random.random([1, 240, 320]) * 255
    # start_time = time.time()
    # image = cv2.resize(image, (240, 752))
    # end_time = time.time()
    # resize_time += end_time - start_time
    input =  np.reshape(image.astype(np.float32), (1, 1, 240, 320))
    # input_data = preprocess_image(image_file_path, target_shape)
    
    start_time = time.time()
    output_data = session.run([output_name1], {input_name: input})
  
    print((output_data[0].shape))
    # image = Image.fromarray(output_data[0].reshape(1, 240, 320).transpose(1, 2, 0))
    # cv2.imshow("1", image)
    # cv2.waitKey(0)
    # cv2.imshow("2", output_data[1])
    # cv2.waitKey(0)
    end_time = time.time()
    total_time += end_time - start_time
print(output_data[0])
print(resize_time / times)
print(total_time / times)
# 输出结果
# print("Output shape:", output_data[0].shape)
# print("Output data:", output_data[0])
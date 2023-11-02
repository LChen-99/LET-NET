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

input_name = session.get_inputs()[0].name
output_name1 = session.get_outputs()[0].name

print(input_name)
print(output_name1)

def nms_fast(in_corners, H, W, dist_thresh):
    """
    Run a faster approximate Non-Max-Suppression on numpy corners shaped:
      3xN [x_i,y_i,conf_i]^T
    Algo summary: Create a grid sized HxW. Assign each corner location a 1, rest
    are zeros. Iterate through all the 1's and convert them either to -1 or 0.
    Suppress points by setting nearby values to 0.
    Grid Value Legend:
    -1 : Kept.
     0 : Empty or suppressed.
     1 : To be processed (converted to either kept or supressed).
    NOTE: The NMS first rounds points to integers, so NMS distance might not
    be exactly dist_thresh. It also assumes points are within image boundaries.
    Inputs
      in_corners - 3xN numpy array with corners [x_i, y_i, confidence_i]^T.
      H - Image height.
      W - Image width.
      dist_thresh - Distance to suppress, measured as an infinty norm distance.
    Returns
      nmsed_corners - 3xN numpy matrix with surviving corners.
      nmsed_inds - N length numpy vector with surviving corner indices.
    """
    grid = np.zeros((H, W)).astype(int)  # Track NMS data.
    inds = np.zeros((H, W)).astype(int)  # Store indices of points.
    # Sort by confidence and round to nearest int.
    inds1 = np.argsort(-in_corners[2, :])
    corners = in_corners[:, inds1]
    rcorners = corners[:2, :].round().astype(int)  # Rounded corners.
    # Check for edge case of 0 or 1 corners.
    if rcorners.shape[1] == 0:
        return np.zeros((3, 0)).astype(int), np.zeros(0).astype(int)
    if rcorners.shape[1] == 1:
        out = np.vstack((rcorners, in_corners[2])).reshape(3, 1)
        return out, np.zeros((1)).astype(int)
    # Initialize the grid.
    for i, rc in enumerate(rcorners.T):
        grid[rcorners[1, i], rcorners[0, i]] = 1
        inds[rcorners[1, i], rcorners[0, i]] = i
    # Pad the border of the grid, so that we can NMS points near the border.
    pad = dist_thresh
    grid = np.pad(grid, ((pad, pad), (pad, pad)), mode='constant')
    # Iterate through points, highest to lowest conf, suppress neighborhood.
    count = 0
    for i, rc in enumerate(rcorners.T):
        # Account for top and left padding.
        pt = (rc[0] + pad, rc[1] + pad)
        if grid[pt[1], pt[0]] == 1:  # If not yet suppressed.
            grid[pt[1] - pad:pt[1] + pad + 1, pt[0] - pad:pt[0] + pad + 1] = 0
            grid[pt[1], pt[0]] = -1
            count += 1
    # Get all surviving -1's and return sorted array of remaining corners.
    keepy, keepx = np.where(grid == -1)
    keepy, keepx = keepy - pad, keepx - pad
    inds_keep = inds[keepy, keepx]
    out = corners[:, inds_keep]
    values = out[-1, :]
    inds2 = np.argsort(-values)
    out = out[:, inds2]
    out_inds = inds1[inds_keep[inds2]]
    return out, out_inds

def getPtsFromHeatmap(heatmap, conf_thresh, nms_dist):
    '''
    :param self:
    :param heatmap:
        np (H, W)
    :return:
    '''

    border_remove = 4

    H, W = heatmap.shape[0], heatmap.shape[1]
    xs, ys = np.where(heatmap >= conf_thresh)  # Confidence threshold.
    sparsemap = (heatmap >= conf_thresh)
    if len(xs) == 0:
        return np.zeros((3, 0))
    pts = np.zeros((3, len(xs)))  # Populate point data sized 3xN.
    pts[0, :] = ys
    pts[1, :] = xs
    pts[2, :] = heatmap[xs, ys]
    pts, _ = nms_fast(pts, H, W, dist_thresh=nms_dist)  # Apply NMS.
    inds = np.argsort(pts[2, :])
    pts = pts[:, inds[::-1]]  # Sort by confidence.
    # Remove points along border.
    bord = border_remove
    toremoveW = np.logical_or(pts[0, :] < bord, pts[0, :] >= (W - bord))
    toremoveH = np.logical_or(pts[1, :] < bord, pts[1, :] >= (H - bord))
    toremove = np.logical_or(toremoveW, toremoveH)
    pts = pts[:, ~toremove]
    return pts
# 定义图像文件路径
# image_file_path = "/home/xreal/xreal/HF/hfnet/doc/demo/db1.jpg"
# 设置目标图像大小，与模型输入的大小相匹配
target_shape = (320, 240)
# 预处理图像
times = 100
total_time = 0
resize_time = 0
for i in range(times):
    image = cv2.imread('/home/caia/lc_ws/LET/imagenet-sample-images/n01440764_tench.JPEG', cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (320, 240))
    image = np.array(image)
    # image = np.random.random([1, 240, 320]) * 255
    input =  np.reshape(image.astype(np.float32), (1, 1, 240, 320))
    start_time = time.time()
    output_data = session.run([output_name1], {input_name: input})
    # print((output_data[0].shape))
    
    pts = getPtsFromHeatmap(output_data[0].squeeze(), 0.010, 3)
    end_time = time.time()
    total_time += end_time - start_time

print(pts)
print(resize_time / times)
print(total_time / times)
# 输出结果
# print("Output shape:", output_data[0].shape)
# print("Output data:", output_data[0])
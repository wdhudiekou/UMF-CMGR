import os
import cv2
import kornia
import torch
import numpy as np


def torch_vsm(img):
    his = torch.zeros(256,  dtype=torch.float32).cuda()
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            his[img[i, j].item()] += 1
    sal = torch.zeros(256, dtype=torch.float32).cuda()
    for i in range(256):
        for j in range(256):
            sal[i] += abs(j - i) * his[j].item()
    map = torch.zeros_like(img, dtype=torch.float32)
    for i in range(256):
        map[torch.where(img == i)] = sal[i]
    if map.max() == 0:
        return torch.zeros_like(img, dtype=torch.float32)
    return map / (map.max())

if __name__ == '__main__':

    ir_path = "../dataset/raw/ctrain/Road/ir"
    vi_path = "../dataset/raw/ctrain/Road/vi"

    ir_file_list = sorted(os.listdir(ir_path))
    vi_file_list = sorted(os.listdir(vi_path))

    ir_map_path = "../dataset/test/ir_map"
    vi_map_path = "../dataset/test/vi_map"

    if not os.path.exists(ir_map_path):
        os.makedirs(ir_map_path)
    if not os.path.exists(vi_map_path):
        os.makedirs(vi_map_path)

    for idx, (ir_filename, vi_filename) in enumerate(zip(ir_file_list, vi_file_list)):

        ir_filepath = os.path.join(ir_path, ir_filename)
        vi_filepath = os.path.join(vi_path, vi_filename)

        img_ir = cv2.imread(ir_filepath, cv2.IMREAD_GRAYSCALE)  # uint8 (256, 256)
        img_vi = cv2.imread(vi_filepath, cv2.IMREAD_GRAYSCALE)  # uint8 (256, 256)

        ir_ts = kornia.utils.image_to_tensor(img_ir).cuda() # torch.Size([1, 256, 256])
        vi_ts = kornia.utils.image_to_tensor(img_vi).cuda() # torch.Size([1, 256, 256])
        map_ir = torch_vsm(ir_ts.squeeze()) # torch.Size([1, 256, 256])
        map_vi = torch_vsm(vi_ts.squeeze()) # torch.Size([1, 256, 256])

        w_ir = 0.5 + 0.5 * (map_ir - map_vi)
        w_vi = 0.5 + 0.5 * (map_vi - map_ir)

        img_w_ir = (kornia.utils.tensor_to_image(w_ir) * 255).astype(np.uint8)
        img_w_vi = (kornia.utils.tensor_to_image(w_vi) * 255).astype(np.uint8)

        ir_save_name = os.path.join(ir_map_path, ir_filename)
        vi_save_name = os.path.join(vi_map_path, vi_filename)

        cv2.imwrite(ir_save_name, img_w_ir)
        cv2.imwrite(vi_save_name, img_w_vi)

        # print(img_w_ir)
        # exit(00)


import sys
import os
import re
import yaml
import numpy as np
import cv2
import json
from pathlib import Path
import open3d as o3d
from collections import defaultdict
from utils.pointcloud_tool import PointCloudUtils
from utils.define_print import DPrint
from utils.image_tool import ImageTool
from camera_info import CamIntrinsics,  SteroCalibInfo
from typing import Type, List, Dict, Tuple, Optional

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

class Metrics:
    """
    Preformation evaluation indicators
    """
    def __init__(self, cam_info:SteroCalibInfo, scale:float=1000.0):
        """
        Init parameters

        Parameters:
            cam_info (SteroCalibInfo): Camera parameter
            scale (float): Scale factor. Defaults to 1000.0.
        """
        self.cam_info = cam_info
        self.scale = scale

    def create_folder(self, folder_path:str) -> None:
        """
        Create folder

        Parameters:
            folder_path (str): folder path

        Returns:
            None
        """  
        if not os.path.isdir(folder_path):
            os.system(f"mkdir -p {folder_path}")

    def get_roi_area_box_and_mask(self, roi_json_path:str, cls_name:list=["box","carton"], delta:list=[0,0,0,0]) -> tuple:
        """
         Extracts bounding boxes and creates a mask image for the specified ROI areas.

        Parameters:
            roi_json_path (str): The path to the JSON file containing object annotations.
            cls_name (list, optional): A list of object categories to extract bounding boxes for. Defaults to ["box", "carton"].
            delta (list, optional): A list of four integers representing the adjustments to the bounding box coordinates.
            The format is [left, top, right, bottom], where each value specifies the number of pixels to adjust.
            Defaults to [0, 0, 0, 0].

        Returns:
            tuple: A tuple containing:
                - A list of bounding boxes (tuples of (x1, y1, x2, y2)).
                - A mask image (2D numpy array) where the ROI areas are filled with white color.
        """
        with open(roi_json_path, 'r') as file:
            data = json.load(file)
        # print(f"data= {data}")
        bboxes = []
        mask_img = np.zeros((self.cam_info.use_height, self.cam_info.use_width), dtype=np.uint8)
        for obj in data.get('objects', []):
            if obj["category"] in cls_name:
                bbox = obj['bbox']
                # Extract the current bounding box coordinates
                x1, y1, x2, y2 = bbox
                # Adjust the bounding box coordinates inward by 10 pixels
                new_x1 = int(x1 + delta[0])
                new_y1 = int(y1 + delta[1])
                new_x2 = int(x2 - delta[2])
                new_y2 = int(y2 - delta[3])
                bboxes.append((new_x1, new_y1, new_x2, new_y2))
                cv2.rectangle(mask_img, (new_x1, new_y1), (new_x2, new_y2), 255, thickness=cv2.FILLED)
        return bboxes, mask_img

    def epe(self, 
            predict_depth_image:np.array, 
            gt_depth_image:np.array,
            depth_max:int=np.inf) -> tuple:
        """
        Computes the End-Point Error (EPE) between predicted and ground-truth depth images.

        Parameters:
            predict_depth_image (np.array): The predicted depth image.
            gt_depth_image (np.array): The ground-truth depth image.
            depth_max (int, optional): The maximum depth value to consider. Defaults to np.inf.

        Returns:
            tuple: A tuple containing:
                - valid_pixel_num (int): The number of valid pixels.
                - depth_diff (np.array): The absolute difference between predicted and ground-truth depth values.
                - disp_diff (np.array): The absolute difference between predicted and ground-truth disparity values.
                - gt_depth_image_mask (np.array): The masked ground-truth depth image.
                - predict_depth_image_mask (np.array): The masked predicted depth image.
        """
        F_B = self.cam_info.F * (self.cam_info.B / self.scale)

        if predict_depth_image.shape != gt_depth_image.shape:
            DPrint.print_color_message(f"Warning: predict_depth_image.shape != gt_depth_image.shape !", "YELLOW")
            height, width = predict_depth_image.shape[:2]
            gt_depth_image = cv2.resize(gt_depth_image, (width, height))

        # 深度值过滤
        gt_depth_image[gt_depth_image > depth_max] = 0
        predict_depth_image[predict_depth_image > depth_max] = 0

        # 获取有效点深度点mask
        valid_mask = np.logical_and(gt_depth_image, predict_depth_image)
        valid_pixel_num = np.count_nonzero(valid_mask)
        if not valid_pixel_num:
            raise ValueError("Error: Depth map does not contain valid points!")

        gt_depth_image[~valid_mask]= 0
        predict_depth_image[~valid_mask] = 0

        gt_disp = np.zeros_like(gt_depth_image, np.float32)
        predict_disp = np.zeros_like(predict_depth_image, np.float32)

        gt_disp[valid_mask] = F_B / (gt_depth_image[valid_mask] / self.scale)
        predict_disp[valid_mask] = F_B / (predict_depth_image[valid_mask] / self.scale)

        # 计算深度图视差epe
        disp_diff = np.abs(gt_disp - predict_disp)

        # 计算深度图的深度绝对值，这里注意书写方式，self.scale写在外面计算结果不一样
        depth_diff = np.abs(gt_depth_image / self.scale - predict_depth_image / self.scale)
       
        gt_depth_image_mask = np.abs(gt_depth_image / self.scale)
        predict_depth_image_mask = np.abs(predict_depth_image / self.scale)

        return valid_pixel_num, depth_diff, disp_diff, gt_depth_image_mask, predict_depth_image_mask

    def epe_and_rmse_roi(self, 
                    predict_depth_dir:str, 
                    gt_depth_dir:str, 
                    image_dir:str,
                    json_dir:str,
                    crop_dir:str,
                    class_name:list,
                    roi_bbox:list=[],
                    coord_offset:list=[],
                    crop_visual:bool=True,
                    depth_max:int=np.inf) -> dict:
        """
        This function calculates the depth and disparity errors between predicted depth maps and ground truth depth maps within specified Regions of Interest (ROIs

        Parameters:
            predict_depth_dir (str): Directory containing predicted depth images (filenames must contain "depth").
            gt_depth_dir (str): Directory containing ground-truth depth images.
            image_dir (str): Directory containing corresponding color images (filenames must contain "left").
            json_dir (str): Directory containing JSON annotation files for automatic ROI detection.
            crop_dir (str): Output directory for cropped ROI visualizations.
            class_name (list): List of class names to detect in JSON annotations.
            roi_bbox (list, optional): Manual ROI coordinates [x1,y1,x2,y2]. Defaults to [] (auto-detection).
            coord_offset (list, optional): Coordinate offsets [dx,dy] to adjust detected ROIs. Defaults to [].
            crop_visual (bool, optional): Whether to save cropped visualizations. Defaults to True.
            depth_max (int, optional): Maximum valid depth value (in mm). Defaults to np.inf.

        Returns:
            dict: A dictionary containing:
                - image_num (int): Total processed image count
                - valid_pixel_num (int): Cumulative valid pixels across all images
                - depth_diff (float): Sum of absolute depth errors
                - disp_diff (float): Sum of disparity EPE errors
                - single_depth_abs (list): Per-image mean depth errors
                - single_disp_epe (list): Per-image mean EPE errors
                - depth_abs (dict): Depth error maps per image {filename: error_map}
                - disp_abs (dict): Disparity error maps per image {filename: error_map}
                - gt_depth (dict): Cropped ground-truth depth maps {filename: depth_map}
                - predict_depth (dict): Cropped predicted depth maps {filename: depth_map}
        """
        info  = {}
        # 图片数量
        info["image_num"] = 0
        # 累计有效点数量
        info["valid_pixel_num"] = 0
        # 累计深度绝对偏差值
        info["depth_diff"] = 0
        # 累计视差epe偏差
        info["disp_diff"] = 0
        # 存放单张图片的深度绝对偏差
        info["single_depth_abs"] = []
        # 存放单张图片的epe偏差
        info["single_disp_epe"] = []
        info["depth_abs"] = {}
        # 存放每张图片名 以及 视差epe偏差图
        info["disp_abs"] = {}

        info["gt_depth"] = {}
        info["predict_depth"] = {}

        self.create_folder(crop_dir)
        # 遍历预测深度图
        for filename in os.listdir(predict_depth_dir):
            if "depth" in filename:
                name = Path(filename).stem
                predict_depth_image_path = os.path.join(predict_depth_dir, filename)
                gt_depth_image_path = os.path.join(gt_depth_dir, filename)
                color_image_path = os.path.join(image_dir, filename.replace("depth", "left"))

                 # 获取roi区域的坐标以及mask图
                if not roi_bbox and os.path.isdir(json_dir):
                    # 如果只有单个检测文件:比如整个数据集使用单个json文件
                    if len(os.listdir(json_dir)) == 1:
                        roi_json_name = os.listdir(json_dir)[0]
                        roi_json_name = os.listdir(json_dir)[0]
                    else:
                        translation_table = str.maketrans({'depth': 'left', 'png': 'json'})
                        roi_json_name = filename.translate(translation_table)

                    roi_json_path = os.path.join(json_dir, roi_json_name)
                    _, mask_img = self.get_roi_area_box_and_mask(roi_json_path,
                                                                cls_name=class_name,
                                                                delta=coord_offset)
                else:
                    mask_img = np.zeros((self.cam_info.use_height, self.cam_info.use_width), dtype=np.uint8)
                    cv2.rectangle(mask_img, (roi_bbox[0], roi_bbox[1]), (roi_bbox[2], roi_bbox[3]), 255, thickness=cv2.FILLED)

                if os.path.exists(gt_depth_image_path) and os.path.exists(predict_depth_image_path) and os.path.exists(color_image_path):
                    info["image_num"] += 1
                    predict_depth_image = cv2.imread(predict_depth_image_path, cv2.IMREAD_UNCHANGED)
                    gt_depth_image = cv2.imread(gt_depth_image_path, cv2.IMREAD_UNCHANGED)
                    color_image = cv2.imread(color_image_path)

                    predict_depth_image = cv2.bitwise_and(predict_depth_image, predict_depth_image, mask=mask_img)
                    gt_depth_image = cv2.bitwise_and(gt_depth_image, gt_depth_image, mask=mask_img)
                    color_image = cv2.bitwise_and(color_image, color_image, mask=mask_img)

                    crop_image_path = os.path.join(crop_dir, Path(color_image_path).name)
                    cv2.imwrite(crop_image_path, color_image)

                    if crop_visual:
                        crop_predict_depth_path = os.path.join(crop_dir, f"{name}_predict.png")
                        crop_gt_depth_path = os.path.join(crop_dir, f"{name}_gt.png")
                        crop_ply_path = os.path.join(crop_dir, f"{name}.ply")
                        cv2.imwrite(crop_predict_depth_path, predict_depth_image)
                        cv2.imwrite(crop_gt_depth_path, gt_depth_image)

                        gt_point_cloud = PointCloudUtils.depth_map_to_point_cloud(gt_depth_image,
                                                                                fx=self.cam_info.intr[0][0],
                                                                                cx=self.cam_info.intr[0][2],
                                                                                fy=self.cam_info.intr[1][1],
                                                                                cy=self.cam_info.intr[1][2],
                                                                                color_map=color_image)
                        if not len(gt_point_cloud.points):
                            DPrint.print_color_message(f"Warning:The point cloud obtained in the {name} cropping depth area is emply! ", "YELLOW")
                            continue
                        
                        o3d.io.write_point_cloud(crop_ply_path, gt_point_cloud)

                    valid_pixel_num, depth_diff, disp_diff, gt_depth_image_mask, predict_depth_image_mask = self.epe(predict_depth_image, 
                                                                                                                    gt_depth_image, 
                                                                                                                    depth_max)

                    info["valid_pixel_num"] += valid_pixel_num

                    sum_depth = np.sum(depth_diff)
                    info["depth_diff"] += sum_depth
                    info["single_depth_abs"].append(sum_depth / valid_pixel_num)

                    sum_disp = np.sum(disp_diff)
                    info["disp_diff"] += sum_disp
                    info["single_disp_epe"].append(sum_disp / valid_pixel_num)

                    color_name = name.replace("depth", "left")
                    info["depth_abs"][color_name] = depth_diff
                    info["disp_abs"][color_name] = disp_diff
                    info["gt_depth"][color_name] = gt_depth_image_mask
                    info["predict_depth"][color_name] =  predict_depth_image_mask
                else:
                    print(f"warning: {gt_depth_image_path}, {predict_depth_image_path}, or {color_image_path} not exists, skip!")
                    continue
        return info

    def fill_rate(self):
        pass


    def temporal_noise(self):
        pass
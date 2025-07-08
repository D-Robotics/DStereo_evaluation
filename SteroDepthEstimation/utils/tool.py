import sys
import os
import yaml
import numpy as np
import cv2
import json
import time
import shutil
from pathlib import Path
import open3d as o3d
from SteroDepthEstimation.utils.pointcloud_tool import PointCloudUtils
from SteroDepthEstimation.utils.image_tool import ImageTool
from SteroDepthEstimation.utils.define_print import DPrint
from camera_info import CamIntrinsics, SteroCalibInfo
from typing import Type, List, Dict, Tuple, Optional

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

class Tools:
    def __init__(self, camera_info:SteroCalibInfo, depth_min:float=0.0, depth_max:float=5.0, sclae:int=1000):
        """
        init of tools class

        Parameters:
            camera_info (CamIntrinsics): camera parameter
            depth_min (float): min value of the depth
            depth_max (float): max value of the depth
            scale (int): scale factor

        Returns:
            depth image and valid point (tuple)
        """
        self.fx = camera_info.intr[0][0]
        self.cx = camera_info.intr[0][2]
        self.fy = camera_info.intr[1][1]
        self.cy = camera_info.intr[1][2]
        self.width = camera_info.use_width
        self.height = camera_info.use_height
        self.LindarToCam = camera_info.LindarToCam
        self.depth_min = depth_min
        self.depth_max = depth_max
        self.scale = 1000

    @classmethod
    def prepare_directory(cls, dir_path:str, clear_history:bool=True) -> None:
        """
        create folder and clear hostory file.

        Parameters:
            dir_path (str): folder of collection datasets.
            clear_history (bool) : clear history data. Defaults to True.

        Returns:
            None
        """
        if os.path.isdir(dir_path):
            if clear_history:
                os.system(f"rm -rf {dir_path}")
                os.system(f"mkdir -p {dir_path}")
        else:
            os.system(f"mkdir -p {dir_path}")

    @classmethod
    def is_pixel_in_bounds(cls, x:int, y:int, image_width:int, image_height:int) -> bool:
        """
        judge whether the point is within the image coordinate.

        Parameters:
            x (int) : x coordinate.
            y (int) : y coordinate.
            image_width (int): image width.
            image_height (int): image height.

        Returns:
            bool
        """
        return (0 <= x < image_width) and (0 <= y < image_height)


    def stereo_lidar_to_camera(self,
                            point_cloud_dir:str, 
                            image_dir:str,
                            save_dir:str,
                            visual_valid_point_dir, 
                            use_transform:bool=True) -> None:
        """
        Batch point cloud transformation from LiDAR coordinate system to image coordinate system and convert depth image.

        Parameters:
            point_cloud_dir (str): Folder of lidar point cloud .
            image_dir (str): Folder of color image.
            save_dir (str): Folder of save depth map.
            visual_valid_point_dir (str): Folder of visual depth image valid point.
            use_transform (bool): whether to use coordinate system conversion. Defaults to True.

        Returns:
            None.
        """
        DPrint.print_color_message("PointCloud to Depth Start", "BLUE")
        if visual_valid_point_dir:
            self.prepare_directory(visual_valid_point_dir)
        for filename in os.listdir(point_cloud_dir):
            # if filename.endswith((".pcd", "ply")):
            if filename.endswith((".pcd")):
                name = Path(filename).stem
                pcd_path = os.path.join(point_cloud_dir, filename)
                image_path = os.path.join(image_dir, f"left{name}.png")
                pcd = o3d.io.read_point_cloud(pcd_path)
                if use_transform:
                    pcd.transform(self.LindarToCam)
                points = np.asarray(pcd.points)
                depth_map, valid_point = PointCloudUtils.point_cloud_to_depth_map(points, 
                                        self.width, 
                                        self.height, 
                                        self.fx, 
                                        self.fy, 
                                        self.cx,
                                        self.cy, 
                                        self.scale, 
                                        self.depth_min, 
                                        self.depth_max
                                    )
                depth_save_path = os.path.join(save_dir, f"depth{name}.png")
                cv2.imwrite(depth_save_path, depth_map) 
                # visual
                if visual_valid_point_dir:
                    if os.path.exists(image_path):
                        color_image = cv2.imread(image_path)
                        for u, v in valid_point:
                            cv2.circle(color_image, (u, v), radius=1, color=(0, 255, 0), thickness=-1)
                        cv2.imwrite(os.path.join(visual_valid_point_dir, f"left{name}.png"), color_image)
        DPrint.print_color_message("PointCloud to Depth End", "BLUE")


    def depth_point_cloud(self, depth_img_dir:str, image_dir:str, save_dir:str, suffix:str="gt") -> None:
            """
            Visual of convert depth map to point cloud.

            Parameters:
                depth_img_dir (str): Folder of depth map.
                image_dir (str): Folder of color image.
                save_dir (str): Save folder of point cloud.
                suffix (str): file suffix.

            Returns:
                None.
            """    
            self.prepare_directory(save_dir)
            for filename in os.listdir(depth_img_dir):
                if "depth" in filename:
                    name = Path(filename).stem
                    gt_depth_image_path = os.path.join(depth_img_dir, filename)
                    color_image_path = os.path.join(image_dir, filename.replace("depth", "left")) 
                    if os.path.exists(color_image_path):
                        depth_image = cv2.imread(gt_depth_image_path, cv2.IMREAD_UNCHANGED)
                        color_image = cv2.imread(color_image_path)
                        point_cloud = PointCloudUtils.depth_map_to_point_cloud(depth_image, 
                                                                                self.fx, 
                                                                                self.fy, 
                                                                                self.cx, 
                                                                                self.cy, 
                                                                                self.scale, 
                                                                                color_map=color_image
                                                                                )

                        ply_path = os.path.join(save_dir, f"{name}_{suffix}.ply")
                        o3d.io.write_point_cloud(ply_path, point_cloud)

    def split_image(self, input_dir:str, output_dir:str) -> None:
        """
        get left and right image by split combined image

        Parameters:
            input_dir (str): folder of the input image
            output_dir (str): folder of the output image

        Returns:
            Tuple
        """
        DPrint.print_color_message("Split Combined Image Start", "BLUE")
        self.prepare_directory(output_dir)
        for filename in os.listdir(input_dir):
            if filename.lower().endswith(('.png', "jpg", "jpeg")):
                name = Path(filename).stem
                img_path = os.path.join(input_dir, filename)
                combined_img = cv2.imread(img_path)
                if combined_img is None:
                    DPrint.print_color_message(f"Warning: read combined image failed {filename}!, skip", "YELLOW")
                    continue
                height, width, _ = combined_img.shape
                left_img = combined_img[:height//2, :]  
                right_img = combined_img[height//2:, :]  
                left_path = os.path.join(output_dir, f"{name}_left.png")
                right_path = os.path.join(output_dir, f"{name}_right.png")
                cv2.imwrite(left_path, left_img)
                cv2.imwrite(right_path, right_img)
        DPrint.print_color_message("Split Combined Image End", "BLUE")

    def bin_to_png(self, input_dir:str, output_dir:str, width:int=1280, height:int=2176) -> None:
        """
        covert bin to png format 

        Parameters:
            input_dir (str): folder of the input image
            output_dir (str): folder of the output image
            width (int): image width 
            height (int): image height

        Returns:
            None
        """
        DPrint.print_color_message("NV12 to PNG Start", "BLUE")
        self.prepare_directory(output_dir, False)
        for filename in os.listdir(input_dir):
            if filename.endswith(".bin"):
                nv12_path = os.path.join(input_dir, filename)
                png_path = os.path.join(output_dir, filename.replace(".bin", ".png"))
                with open(nv12_path, 'rb') as f:
                    nv12_data = f.read()
                y_size = width * height
                total_size = width * height + width * height // 2
                if len(nv12_data) != total_size:
                    pcd_path = nv12_path.replace(".bin", ".pcd")
                    DPrint.print_color_message(f"NV12 size error, target={total_size}, real={len(nv12_data)}", "RED")
                    os.system(f"rm -rf {nv12_path} {pcd_path}")
                    continue
                uv_size = y_size // 2
                y_plane = np.frombuffer(nv12_data[:y_size], dtype=np.uint8).reshape((height, width))
                uv_plane = np.frombuffer(nv12_data[y_size:y_size + uv_size], dtype=np.uint8).reshape((height // 2, width))
                yuv_img = np.vstack((y_plane, uv_plane))
                bgr_img = cv2.cvtColor(yuv_img, cv2.COLOR_YUV2BGR_NV12)
                cv2.imwrite(png_path, bgr_img)
        DPrint.print_color_message("NV12 to PNG End", "BLUE")


    def image_remap(self, input_dir:str, camera_info:SteroCalibInfo, output_dir:str) -> None:
        """
        image remapping correction 

        Parameters:
            input_folder (str): folder of the input image
            camera_info (SteroCalibInfo): camer parameter 
            output_dir (str): folder of the output image

        Returns:
            None
        """
        DPrint.print_color_message("Image Rectify Start", "BLUE")
        self.prepare_directory(output_dir)
        left_img_filepaths = []
        right_img_filepaths = []
        for filename in os.listdir(input_dir):
            if 'left' in filename:
                left_img_filepaths.append(filename)
            if 'right' in filename:
                right_img_filepaths.append(filename)
        print(f"left_img_filepaths num = {len(left_img_filepaths)}")
        print(f"right_img_filepaths num = {len(right_img_filepaths)}")
        assert len(left_img_filepaths) == len(right_img_filepaths), "len(left_img_filepaths) =! len(right_img_filepaths)"

        for filename in left_img_filepaths:
            name = filename.split("_")[0]
            filename_l = os.path.join(input_dir, filename)
            filename_r = filename_l.replace("left", "right")
            img_l_rectified, img_r_rectified = camera_info.rectify_img(filename_l, filename_r)
            cv2.imwrite(os.path.join(output_dir, f'left{name}.png'), img_l_rectified)
            cv2.imwrite(os.path.join(output_dir, f'right{name}.png'), img_r_rectified)
        DPrint.print_color_message("Image Rectify End", "BLUE")

    def rename(self, input_dir:str, output_dir:str) -> None:
        """
        File rename

        Parameters:
            input_dir (str): folder of the input image
            output_dir (str): folder of the output image

        Returns:
            None
        """
        DPrint.print_color_message("Image Rename Start", "BLUE")
        self.prepare_directory(output_dir, False)
        file_counter = 0
        for filename in os.listdir(input_dir):
            if "left" in filename:
                left_image_path = os.path.join(input_dir, filename)
                right_name_path = left_image_path.replace("left", "right")
                depth_name_path = left_image_path.replace("left", "depth")
                if not os.path.exists(right_name_path):
                    DPrint.print_color_message(f"Warning: {right_name_path} not exists, skip !", "YELLOW")
                    continue
                if not os.path.exists(depth_name_path):
                    DPrint.print_color_message(f"Warning: {depth_name_path} not exists, skip !", "YELLOW")
                    continue
                left_new_filename = f"left{file_counter:06d}.png" 
                right_new_filename =  f"right{file_counter:06d}.png" 
                depth_new_filename =  f"depth{file_counter:06d}.png" 
                left_image_destination_path = os.path.join(output_dir, left_new_filename)
                right_image_destination_path = os.path.join(output_dir, right_new_filename)
                depth_image_destination_path = os.path.join(output_dir, depth_new_filename)
                os.system(f"mv {left_image_path} {left_image_destination_path}")
                os.system(f"mv {right_name_path} {right_image_destination_path}")
                os.system(f"mv {depth_name_path} {depth_image_destination_path}")
                file_counter += 1
        # print(f"file_counter = {file_counter}")
        DPrint.print_color_message("Image Rename End", "BLUE")


    def datastes_combine(self, input_dir_list:dict, output_dir:str) -> None:
        """
        Datastes combine to a folder

        Parameters:
            input_dir_list (dict)): list of input image folder
            output_dir (str): folder of the output image

        Returns:
            None
        """
        DPrint.print_color_message("Image Remove Start", "BLUE")
        self.prepare_directory(output_dir)
        for folder in input_dir_list:
            for filename in os.listdir(folder):
                source_path = os.path.join(folder, filename)
                destination_path = os.path.join(output_dir, filename)
                shutil.copy(source_path, destination_path)
        DPrint.print_color_message("Image Remove End", "BLUE")

    
    def generate_fixed_depth_map(self, input_dir:str, save_dir:str, width:int, height:str, distance_mm:int) -> None:
        """
         Generates a depth map with const depth value

        Parameters:
            input_dir (str): The foldar of the images.
            save_dir (str):  The foldar of save depth map.
            width (int): The width of the depth map in pixels.
            height (int): The height of the depth map in pixels.
            distance_mm (float): The fixed depth value in millimeters.

        Returns:
            None
        """
        for filename in os.listdir(input_dir):
            if "left" in filename:
                name = Path(filename).stem
                gt_dapth_image_path = os.path.join(save_dir,  filename.replace("left", "depth"))
                depth_map = ImageTool.generate_fixed_depth_map(width, height, distance_mm)
                cv2.imwrite(gt_dapth_image_path, depth_map) 

    def get_roi_area(self, json_dir:str, delta:int=10) -> dict:
        """
        Get roi area coordiation of json path

        Parameters:
            json_dir (str): The foldar of inference info.
            delta (int):  The coordination offset.

        Returns:
            dict
        """
        bbox_info = {}
        for filename in os.listdir(json_dir):
            datasets_dir_name = Path(filename).stem
            json_file_path = os.path.join(json_dir, filename)
            with open(json_file_path, 'r') as file:
                data = json.load(file)

            adjusted_bboxes = []
            box = [0, 0, 0, 0]
            for obj in data.get('objects', []):
                if obj.get("category")  not in ["panda"]:
                    bbox = obj.get('bbox')
                    if bbox:
                        # Extract the current bounding box coordinates
                        x1, y1, x2, y2 = bbox
                        # Adjust the bounding box coordinates inward by 10 pixels
                        new_x1 = int(x1 + delta)
                        new_y1 = int(y1 + delta)
                        new_x2 = int(x2 - delta)
                        new_y2 = int(y2 - delta)
                        adjusted_bboxes.append([new_x1, new_y1, new_x2, new_y2])
                else:
                    continue
            if len(adjusted_bboxes):
                box = adjusted_bboxes[0]
            bbox_info[datasets_dir_name] = box

        return bbox_info
import os
import sys
import time
import yaml
import shutil
import cv2
import re
from pathlib import Path
import argparse
import numpy as np
from SteroDepthEstimation.utils.tool import Tools
from SteroDepthEstimation.utils.sensor_synchronizer import SensorSynch
from SteroDepthEstimation.utils.visual import Visual
from SteroDepthEstimation.utils.image_tool import ImageTool
from SteroDepthEstimation.utils.pointcloud_tool import PointCloudUtils
from SteroDepthEstimation.utils.define_print import DPrint
from metrics.metrics import Metrics
import multiprocessing
import subprocess
from collections import defaultdict
import open3d as o3d
import camera_info as cam_info

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

class SteroDepthEstimation:
    """
    Performance evaluation of binocular depth estimation algorithm
    """
    datatsets_save_dir = defaultdict(dict)

    def __init__(self, args:argparse.Namespace):
        self.args = args
        for arg_name, arg_value in vars(self.args).items():
            print(f"{arg_name}=: {arg_value}")
        [self.create_folder(dataset_name) for dataset_name in os.listdir(self.args.datasets_root_dir)]
        self.stero_calib_info = self.parse_camera_info
        self.max_processes = multiprocessing.cpu_count()
        self.debug = self.args.debug
        self.tools = Tools(self.stero_calib_info, self.args.depth_min, self.args.depth_max)
        self.rectify_dataset_path = []
        self.metrics_result = []
        self.class_name = self.args.use_category
        self.use_fixed_distance = True
        self.metrics = Metrics(self.stero_calib_info)
        self.visual = Visual()

    @property
    def parse_camera_info(self):
        """
        Parese json file

        Parameters:
            None

        Returns:
            List: List of parsed data
        """
        stero_calib_info = cam_info.SteroCalibInfo(self.args.camera_info_path)
        return stero_calib_info

    def create_folder(self, name:str) -> None:
        """
        Create folder

        Parameters:
            name (str): Dataset name

        Returns:
            None
        """
        save_dir = os.path.join(self.args.datasets_root_dir, name, "result")
        SteroDepthEstimation.datatsets_save_dir[name]["save_dir"] = save_dir
        SteroDepthEstimation.datatsets_save_dir[name]["images"] =  os.path.join(save_dir, "images")
        SteroDepthEstimation.datatsets_save_dir[name]["gt_lidar"] =  os.path.join(save_dir, "gt_lidar")
        SteroDepthEstimation.datatsets_save_dir[name]["predict"] =  os.path.join(save_dir, "predict")
        SteroDepthEstimation.datatsets_save_dir[name]["visual"] =  os.path.join(save_dir, "visual")
        SteroDepthEstimation.datatsets_save_dir[name]["ply"] =  os.path.join(save_dir, "ply")
        SteroDepthEstimation.datatsets_save_dir[name]["crop"] =  os.path.join(save_dir, "crop")

        SteroDepthEstimation.datatsets_save_dir[name]["predict_foundation"] = os.path.join(save_dir, "predict_foundation")
        SteroDepthEstimation.datatsets_save_dir[name]["foundation_valid_point"] = os.path.join(save_dir, "foundation_valid_point")
        SteroDepthEstimation.datatsets_save_dir[name]["gt_foundation"] = os.path.join(save_dir, "gt_foundation")
        SteroDepthEstimation.datatsets_save_dir[name]["visual_gt_foundation"] = os.path.join(save_dir, "visual_gt_foundation")

        SteroDepthEstimation.datatsets_save_dir[name]["roi_detect_visual"] = os.path.join(save_dir, "roi_area", "roi_area_visual")
        SteroDepthEstimation.datatsets_save_dir[name]["roi_detect_json"] = os.path.join(save_dir, "roi_area", "roi_area_json")
        
    def foundation_process(self, name:str) -> subprocess.Popen:
        """
        Runs the foundation model prediction for a given dataset

        Parameters:
            name (str): The name of the dataset for which the prediction is to be made.

        Returns:
            subprocess.Popen: The subprocess object representing the started process.
        """
        import subprocess
        params = {
            "--img_dir": SteroDepthEstimation.datatsets_save_dir[name]["images"],
            "--out_dir": SteroDepthEstimation.datatsets_save_dir[name]["predict_fundation"]
        }
        command = ['python', './FoundationStereo/scripts/run_demo.py']
        for key, value in params.items():
            command.append(f"{key}={value}")

        # 启动子进程，不阻塞主程序
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        DPrint.print_color_message(f"Start sub-process ID = {process.pid}", "RED")
        return process


    def depth_gt_foundation(self, inference:bool=False) -> None:
        """
        Processes all datasets using multiple processes.

        Parameters:
            inference (bool, optional): Whether to perform foundation stereo model inference. Defaults to False.

        Returns:
            None
        """
        for name in SteroDepthEstimation.datatsets_save_dir.keys():
            if inference:
                DPrint.print_color_message(f"USE Foundationc Predict", "BLUE")
                if not os.path.isdir(SteroDepthEstimation.datatsets_save_dir[name]["images"]):
                    DPrint.print_color_message(f"{name} not exists images folder", "RED")
                    continue
                sub_process = self.foundation_process(name)
                stdout, stderr = sub_process.communicate()
                print("sub-process output info = :", stdout)
                print("sub-process output errro info = :", stderr)

                # 定期检查子进程是否结束
                while True:
                    ret = sub_process.poll()
                    if ret is not None:
                        DPrint.print_color_message(f"Sub-process (ID = {sub_process.pid}) has finished with return code: {ret}", "GREEN")
                        stdout, stderr = sub_process.communicate()
                        DPrint.print_color_message(f"Sub-process output:\n{stdout.decode()}", "BLUE")
                        if stderr:
                            DPrint.print_color_message(f"Sub-process errors:\n{stderr.decode()}", "RED")
                            return
                        break
                    DPrint.print_color_message("Sub-process is still running...", "BLUE")
                    time.sleep(1)
            self.post_precess_foundataion(name)

    def post_precess_foundataion(self, name:str, debug:bool=True) -> None:
        """
        Performs post-processing for the foundation model prediction results.

        Parameters:
            name (str): The name of the dataset.
            debug (bool, optional): Whether to enable debug mode. Defaults to True.

        Returns:
            None
        """
        name = SteroDepthEstimation.datatsets_save_dir[name]
        if not os.path.isdir(name["predict_fundation"]):
            DPrint.print_color_message(f"{name} not exists images folder", "RED")
            return
        for filename in os.listdir(name["predict_fundation"]):
            if ".ply" in filename:
                file_path = os.path.join(name["predict_fundation"], filename)
                # translation_table = str.maketrans({'.png': '', 'cloud_left': ''})
                # name = filename.translate(translation_table)
                file_name = filename.replace(".png", "").replace("cloud_left", "")
                new_file_path = os.path.join(name["predict_fundation"], file_name)
                os.system(f"mv {file_path} {new_file_path}")

        self.tools.stero_lidar_to_camera(name["predict_fundation"], 
                                        name["images"],
                                        name["images"],
                                        name["fundation_valid_point"], 
                                        use_transform=False,
                                        debug=debug)

        Tools.prepare_directory(name["gt_fundation"])
        src_path = os.path.join(name["images"], "depth*")
        dst_path = name["gt_fundation"]
        os.system(f"mv {src_path} {dst_path}")

    def depth_gt(self) -> None:
        """
        Processes depth ground truth for all datasets in the specified root directory. 
        It then processes each dataset using multiple processes if there are multiple data points in the dataset.

        Parameters:
            None

        Returns:
            None
        """
        for dataset_name in os.listdir(self.args.datasets_root_dir):
            DPrint.print_color_message(f"Start of process {dataset_name} ", "YELLOW")
            # 不定距的数据集命名使用格式："tile_0", 定距离的数据集使用格式："tile_100"
            name, depth_value = dataset_name.split("_")
            # if dataset_name == "box_80":
            #     continue
            datasets_dir = os.path.join(self.args.datasets_root_dir, dataset_name)
            # 清除历史数据
            if not self.args.use_history_data:
                os.system(f"rm -rf {datasets_dir}/result")
                # os.system(f"rm -rf {datasets_dir}/../temp")
                # continue
            datasets_path = [os.path.join(datasets_dir, d) for d in os.listdir(datasets_dir) if os.path.isdir(os.path.join(datasets_dir, d))]
            print(f"datasets_path={datasets_path}")

            use_processes = self.max_processes if len(datasets_path) > self.max_processes else len(datasets_path)
            print(f"max_processes={self.max_processes} , use_processes={use_processes}")

            self.rectify_dataset_path.clear()
            # 当每个数据集中有多个数据时候使用多进程
            with multiprocessing.Pool(processes=self.max_processes) as pool:
                for dataset_path in datasets_path:
                    pool.apply_async(self.pre_process, args=(name, int(depth_value), dataset_path), callback=self.collect_gt_results)
                pool.close()
                pool.join()
            self.post_process(dataset_name)
            DPrint.print_color_message(f"End of process {dataset_name}", "YELLOW")

    def pre_process(self, name:str, depth_vale:int, data_path:str, scale_factor:int=10) -> str:
        """
        Performs pre-processing for a dataset.

        Parameters:
            name (str): The name of the dataset.
            depth_value (int): The depth value to use if fixed depth is enabled.
            data_path (str): The path to the dataset directory.
            scale_factor (int, optional): The scale factor for depth values. Defaults to 10.

        Returns:
            str: The path to the rectified image and depth directory.
        """
        if os.path.isdir(data_path):
            image_dir = os.path.join(data_path, "image")
            lidar_dir = os.path.join(data_path, "pcd")
            if (not Path(image_dir).is_dir()) or (not Path(lidar_dir).is_dir()):
                DPrint.print_color_message(f"Warning: Folder {data_path} must contain folder image and pcd !", "RED")
                return
            # 存放时间同步数据
            time_syn_dir = os.path.join(data_path, "temp", "syn_data")
            # 存放分开的左右图数据
            left_right_image_dir = os.path.join(data_path, "temp", "split_image")
            # 存放矫正数据
            rectify_image_and_depth_dir = os.path.join(data_path, "temp", "rectify_image")
            # 存放在点云映射到image的图
            depth_valid_point_dir = os.path.join(data_path, "temp", "visual_depth_valid_point")

            # 时间同步
            sensor_syn = SensorSynch(self.args.time_window, self.args.time_threshold)
            sensor_syn.align_sensor_data(image_dir, lidar_dir, time_syn_dir)
            # bin转png
            self.tools.bin_to_png(time_syn_dir, time_syn_dir, self.args.bin_width, self.args.bin_height)
            # 拆分左右图 
            self.tools.split_image(time_syn_dir, left_right_image_dir)
            # 畸变矫正
            self.tools.image_remap(left_right_image_dir, self.stero_calib_info, rectify_image_and_depth_dir)

            # pointcloud转depth
            if depth_vale and self.args.use_fixed_depth:
                DPrint.print_color_message(f"use fixed depth", "BLUE")
                self.tools.generate_fixed_depth_map(rectify_image_and_depth_dir, 
                                                    rectify_image_and_depth_dir,
                                                    self.stero_calib_info.use_width,
                                                    self.stero_calib_info.use_height, 
                                                    depth_vale * scale_factor) 
            else:
                DPrint.print_color_message(f"use PointCloud to depth", "BLUE")
                # 将转换的depth保存在rectify_image_dir路径下,方便查看对应关系
                self.tools.stereo_lidar_to_camera(time_syn_dir, 
                                                rectify_image_and_depth_dir,
                                                rectify_image_and_depth_dir, 
                                                depth_valid_point_dir)
            return rectify_image_and_depth_dir

    def post_process(self, name:str) -> None:
        """
        Performs post-processing for a dataset.

        Parameters:
            name (str): The name of the dataset.

        Returns:
            None
        """
        name = SteroDepthEstimation.datatsets_save_dir[name]
        # 将各个子数据集的结果进行合并
        self.tools.datastes_combine(self.rectify_dataset_path, name["images"])
        # 数据重新命名，用于后面输入模型进行推理获得预测深度图或者视差图（模型推理规定了数据必须使用特定格式,所以必须这么做)
        self.tools.rename(name["images"],  name["images"])
        # 将GT-depth单独存放再一个文件中
        Tools.prepare_directory(name["gt_lidar"])

        for filename in os.listdir(name["images"]):
            if "depth" in filename:
                source_path = os.path.join(name["images"], filename)
                destination_path = os.path.join(name["gt_lidar"], filename)
                shutil.move(source_path, destination_path)
        if self.debug:
            self.tools.depth_point_cloud(name["gt_lidar"], 
                                name["images"], 
                                name["ply"], 
                                "lidar")


    def collect_gt_results(self, result:str) -> None:
        """
        Collects the results from each process.

        Parameters::
            result (str): The result from processing a dataset.

        Returns:
            None
        """

        self.rectify_dataset_path.append(result)
        print(f"self.rectify_dataset_path = {self.rectify_dataset_path}")

    def metrics(self) -> None:
        """
        Evaluates the performance metrics for all datasets.

        Parameters:
            None

        Returns:
            None
        """
        metrics = Metrics(self.stero_calib_info)
        visual = Visual()

        distance = []
        epe_error = []
        depth_error = []

        result = {}

        save_path_info = SteroDepthEstimation.datatsets_save_dir
        print(f"datatsets_save_dir = {save_path_info}")

        use_processes = self.max_processes if len(save_path_info) > self.max_processes else len(save_path_info)

        for dataset_name, save_dir in save_path_info.items():
            print(f"Process datasets name = {dataset_name}")
            name, depth_value = dataset_name.split("_")
            result[dataset_name]  = {}        
            if not depth_value:
                self.use_fixed_distance = False

            # gt使用的数据类型
            if self.args.use_foundation:
                label = "gt-foundation"
                gt_dir = save_dir["gt_foundation"]
                visual_dir = save_dir["visual_gt_foundation"]
            else:
                label = "gt-lidar"
                gt_dir = save_dir["gt_lidar"]
                visual_dir = save_dir["visual"]
            
            # 是否使用感兴趣区域
            if self.args.use_roi_area:
                bboxs = []
            else:
                bboxs = [0, 0, self.stero_calib_info.use_width , self.stero_calib_info.use_height]

            info = metrics.epe_and_rmse_roi(save_dir["predict"], 
                                            gt_dir, 
                                            save_dir["images"],
                                            save_dir["roi_detect_json"],
                                            save_dir["crop"],
                                            class_name=self.class_name,
                                            roi_bbox =bboxs,
                                            coord_offset=self.args.coord_offest,
                                            crop_visual=self.debug,
                                            depth_max=self.args.depth_max)
            image_num = info["image_num"]
            toltal_valid_pixel_num = info["valid_pixel_num"]
            single_image_avg_valid_num = int(info["valid_pixel_num"] / info["image_num"])
            depth_abs_error = info["depth_diff"] / info["valid_pixel_num"]
            disp_epe_error = info["disp_diff"] / info["valid_pixel_num"]

            
            if self.use_fixed_distance:
                distance.append(int(depth_value))
                epe_error.append(depth_abs_error)
                depth_error.append(disp_epe_error)

            # DPrint.print_color_message(f"{dataset_name}-Performance Evaluation Result", "RED")
            # DPrint.print_color_message(f"@:Image-num = {image_num}", "RED")
            # DPrint.print_color_message(f"@:Totla-valid-pixel-num = {toltal_valid_pixel_num}", "YELLOW")
            # DPrint.print_color_message(f"@:Image-avg-valid-pixel-num = {single_image_avg_valid_num}", "YELLOW")
            # DPrint.print_color_message(f"@:Depth-ABS-Error = {depth_abs_error}", "RED")
            # DPrint.print_color_message(f"@:Disp-EPE = {disp_epe_error}", "RED")

            result[dataset_name]["Image-num"] = image_num
            result[dataset_name]["Totla-valid-pixel-num"] = toltal_valid_pixel_num
            result[dataset_name]["Image-avg-valid-pixel-num"] = single_image_avg_valid_num
            result[dataset_name]["Depth-ABS-Error"] = depth_abs_error
            result[dataset_name]["Disp-EPE"] = disp_epe_error

            self.tools.prepare_directory(visual_dir, clear_history=False)

            depth_save_path = os.path.join(visual_dir, "depth.png")
            visual.visualize_error_distribution(info["single_depth_abs"], 
                                                0.0,
                                                0.8,
                                                0.01, 
                                                title="depth-distribution-interval", 
                                                xlabel="error interval(m)", 
                                                ylabel="ratio", 
                                                save_name=depth_save_path)

            disp_save_path = os.path.join(visual_dir, "disp.png")
            visual.visualize_error_distribution(info["single_disp_epe"], 
                                                0, 
                                                15, 
                                                0.2, 
                                                title="disp-epe-distribution-interval", 
                                                xlabel="error interval(m)", 
                                                ylabel="ratio", 
                                                save_name=disp_save_path)

            for name, depth_diff in info["depth_abs"].items():
                units = "(m)"
                color_img_path = os.path.join(save_dir["crop"],  f"{name}.png")
                color_img = cv2.imread(color_img_path)
                save_path_2d = os.path.join(visual_dir, f"{name}_depth_2d.png")
                save_path_3d = os.path.join(visual_dir, f"{name}_depth_3d.png")
                visual.plot_2d_deviation(depth_diff, 
                                        color_img, 
                                        cmap='coolwarm', 
                                        save_path=save_path_2d, 
                                        colorbar=f"value{units}",
                                        title=f"2D-Depth-Error-Visualization")
                visual.plot_3d_deviation(depth_diff, 
                                        color_img, 
                                        cmap='coolwarm', 
                                        save_path=save_path_3d, 
                                        z = "depth-abs-error",
                                        colorbar=f"value{units}",
                                        title=f"3D-Depth-Error-Visualization")
                break

            for name, disp_diff in info["disp_abs"].items():
                units = "(pixel)"
                color_img_path = os.path.join(save_dir["crop"],  f"{name}.png")
                color_img = cv2.imread(color_img_path)
                save_path_2d = os.path.join(visual_dir, f"{name}_disp_2d.png")
                save_path_3d = os.path.join(visual_dir, f"{name}_disp_3d.png")
                visual.plot_2d_deviation(disp_diff, 
                                        color_img, 
                                        cmap='coolwarm', 
                                        save_path=save_path_2d, 
                                        colorbar=f"value{units}",
                                        title=f"2D-Disp-Error-Visualization")
                visual.plot_3d_deviation(disp_diff, 
                                        color_img, 
                                        cmap='coolwarm', 
                                        save_path=save_path_3d,
                                        z="disp-abs-error",
                                        colorbar=f"value{units}", 
                                        title=f"3D-Disp-Error-Visualization")
                break

            for name, depth_diff in info["gt_depth"].items():
                units = "(m)"
                color_img_path = os.path.join(save_dir["crop"],  f"{name}.png")
                color_img = cv2.imread(color_img_path)
                save_path_2d = os.path.join(visual_dir, f"{name}_gt_depth_2d.png")
                save_path_3d = os.path.join(visual_dir, f"{name}_gt_depth_3d.png")
                visual.plot_2d_deviation(depth_diff, 
                                        color_img, 
                                        cmap='coolwarm', 
                                        save_path=save_path_2d, 
                                        colorbar=f"value{units}",
                                        title=f"2D-GT-Depth-Visualization")
                visual.plot_3d_deviation(depth_diff, 
                                        color_img, 
                                        cmap='coolwarm', 
                                        save_path=save_path_3d, 
                                        z="gt-depth-abs",
                                        colorbar=f"value{units}",
                                        title=f"3D-GT-Depth-Visualization")
                break

            for name, depth_diff in info["predict_depth"].items():
                units = "(m)"
                color_img_path = os.path.join(save_dir["crop"],  f"{name}.png")
                color_img = cv2.imread(color_img_path)
                save_path_2d = os.path.join(visual_dir, f"{name}_predict_depth_2d.png")
                save_path_3d = os.path.join(visual_dir, f"{name}_predict_depth_3d.png")
                visual.plot_2d_deviation(depth_diff, 
                                        color_img, 
                                        cmap='coolwarm', 
                                        save_path=save_path_2d, 
                                        colorbar=f"value{units}",
                                        title=f"2D-Predict-Depth-Visualization")
                visual.plot_3d_deviation(depth_diff, 
                                        color_img, 
                                        cmap='coolwarm', 
                                        save_path=save_path_3d,
                                        z="predict-depth-abs",
                                        colorbar=f"value{units}",
                                        title=f"3D-Predict-Depth-Visualization")
                break

        if self.use_fixed_distance:
            print(f"distance = {distance}")
            print(f"depth_error = {depth_error}")
            print(f"epe_error = {epe_error}")
            visual.plot_and_save_curve(distance, depth_error, title= "Depth Curve Plot", xlabel = "ditance", ylabel= "depth_abs_error", output_path=f"./{label}_depth_curve_plot.png")
            visual.plot_and_save_curve(distance, epe_error, title= "EPE Curve Plot", xlabel = "ditance", ylabel= "epe_error", output_path=f"./{label}_epe_curve_plot.png")
        
        DPrint.print_color_message(f"@:result = {result}", "RED")


    def collect_metrics_results(self, result:dict) -> None:
        """
        Collects the results from each evaluate process.

        Parameters::
            result (dict): The result from processing a dataset.

        Returns:
            None
        """

        self.metrics_result.append(result)

    def evaluate(self):
        """
        Evaluates the performance metrics for use multi-process.

        Parameters:
            dataset_name (str): Datasets name.

        Returns:
            None
        """
        save_path_info = SteroDepthEstimation.datatsets_save_dir
        print(f"Datatsets_save_dir = {save_path_info}")
        # use_processes = self.max_processes if len(save_path_info) > self.max_processes else len(save_path_info)
        with multiprocessing.Pool(processes=self.max_processes) as pool:
            for dataset_name in save_path_info.keys():
                print(f"Metrics process datasets name = {dataset_name}")
                pool.apply_async(self.metrics_mutil_process, args=(dataset_name,), callback=self.collect_metrics_results)
            pool.close()
            pool.join()

        DPrint.print_color_message(f"@:result = {self.metrics_result}", "RED")

        distance = []
        depth_error = []
        epe_error = []
        for info in self.metrics_result:
            name, depth_value = info["dataset_name"].split("_")
            if int(depth_value):
                distance.append(int(depth_value))
                depth_error.append(info["Depth-ABS-Error"])
                epe_error.append(info["Disp-EPE"])
        print(f"distance = {distance}")
        print(f"depth_error = {depth_error}")
        print(f"epe_error = {epe_error}")
        self.visual.plot_and_save_curve(distance, depth_error, title= "Depth Curve Plot", xlabel = "ditance", ylabel= "depth_abs_error", output_path=f"./depth_curve_plot.png")
        self.visual.plot_and_save_curve(distance, epe_error, title= "EPE Curve Plot", xlabel = "ditance", ylabel= "epe_error", output_path=f"./epe_curve_plot.png")
        

    def metrics_mutil_process(self, dataset_name:str) -> dict:
        """
        Evaluates the performance metrics for single datasets.

        Parameters:
            dataset_name (str): Datasets name.

        Returns:
            dict
        """
        save_dir = SteroDepthEstimation.datatsets_save_dir[dataset_name]
        name, depth_value = dataset_name.split("_")
        metrics_result= {}
        metrics_result["dataset_name"] = dataset_name       
        # gt使用的数据类型
        if self.args.use_foundation:
            label = "gt-foundation"
            gt_dir = save_dir["gt_foundation"]
            visual_dir = save_dir["visual_gt_foundation"]
        else:
            label = "gt-lidar"
            gt_dir = save_dir["gt_lidar"]
            visual_dir = save_dir["visual"]
        
        # 是否使用感兴趣区域
        if self.args.use_roi_area:
            bboxs = []
        else:
            bboxs = [0, 0, self.stero_calib_info.use_width , self.stero_calib_info.use_height]
        info = self.metrics.epe_and_rmse_roi(save_dir["predict"], 
                                        gt_dir, 
                                        save_dir["images"],
                                        save_dir["roi_detect_json"],
                                        save_dir["crop"],
                                        class_name=self.class_name,
                                        roi_bbox =bboxs,
                                        coord_offset=self.args.coord_offest,
                                        crop_visual=self.debug,
                                        depth_max=int(self.args.depth_max * 1000))
        # print(f"info = {info}")
        metrics_result["Image-num"] = info["image_num"]
        metrics_result["Totla-valid-pixel-num"] = info["valid_pixel_num"]
        metrics_result["Image-avg-valid-pixel-num"] = int(info["valid_pixel_num"] / info["image_num"])
        metrics_result["Depth-ABS-Error"] = info["depth_diff"] / info["valid_pixel_num"]
        metrics_result["Disp-EPE"] = info["disp_diff"] / info["valid_pixel_num"]

        self.tools.prepare_directory(visual_dir, clear_history=False)

        if False:
            depth_save_path = os.path.join(visual_dir, "depth.png")
            self.visual.visualize_error_distribution(info["single_depth_abs"], 
                                                0.0,
                                                0.8,
                                                0.01, 
                                                title="depth-distribution-interval", 
                                                xlabel="error interval(m)", 
                                                ylabel="ratio", 
                                                save_name=depth_save_path)

            disp_save_path = os.path.join(visual_dir, "disp.png")
            self.visual.visualize_error_distribution(info["single_disp_epe"], 
                                                0, 
                                                15, 
                                                0.2, 
                                                title="disp-epe-distribution-interval", 
                                                xlabel="error interval(m)", 
                                                ylabel="ratio", 
                                                save_name=disp_save_path)

            for name, depth_diff in info["depth_abs"].items():
                units = "(m)"
                color_img_path = os.path.join(save_dir["crop"],  f"{name}.png")
                color_img = cv2.imread(color_img_path)
                save_path_2d = os.path.join(visual_dir, f"{name}_depth_2d.png")
                save_path_3d = os.path.join(visual_dir, f"{name}_depth_3d.png")
                self.visual.plot_2d_deviation(depth_diff, 
                                        color_img, 
                                        cmap='coolwarm', 
                                        save_path=save_path_2d, 
                                        colorbar=f"value{units}",
                                        title=f"2D-Depth-Error-Visualization")
                self.visual.plot_3d_deviation(depth_diff, 
                                        color_img, 
                                        cmap='coolwarm', 
                                        save_path=save_path_3d, 
                                        z = "depth-abs-error",
                                        colorbar=f"value{units}",
                                        title=f"3D-Depth-Error-Visualization")
                break

            for name, disp_diff in info["disp_abs"].items():
                units = "(pixel)"
                color_img_path = os.path.join(save_dir["crop"],  f"{name}.png")
                color_img = cv2.imread(color_img_path)
                save_path_2d = os.path.join(visual_dir, f"{name}_disp_2d.png")
                save_path_3d = os.path.join(visual_dir, f"{name}_disp_3d.png")
                self.visual.plot_2d_deviation(disp_diff, 
                                        color_img, 
                                        cmap='coolwarm', 
                                        save_path=save_path_2d, 
                                        colorbar=f"value{units}",
                                        title=f"2D-Disp-Error-Visualization")
                self.visual.plot_3d_deviation(disp_diff, 
                                        color_img, 
                                        cmap='coolwarm', 
                                        save_path=save_path_3d,
                                        z="disp-abs-error",
                                        colorbar=f"value{units}", 
                                        title=f"3D-Disp-Error-Visualization")
                break

            for name, depth_diff in info["gt_depth"].items():
                units = "(m)"
                color_img_path = os.path.join(save_dir["crop"],  f"{name}.png")
                color_img = cv2.imread(color_img_path)
                save_path_2d = os.path.join(visual_dir, f"{name}_gt_depth_2d.png")
                save_path_3d = os.path.join(visual_dir, f"{name}_gt_depth_3d.png")
                self.visual.plot_2d_deviation(depth_diff, 
                                        color_img, 
                                        cmap='coolwarm', 
                                        save_path=save_path_2d, 
                                        colorbar=f"value{units}",
                                        title=f"2D-GT-Depth-Visualization")
                self.visual.plot_3d_deviation(depth_diff, 
                                        color_img, 
                                        cmap='coolwarm', 
                                        save_path=save_path_3d, 
                                        z="gt-depth-abs",
                                        colorbar=f"value{units}",
                                        title=f"3D-GT-Depth-Visualization")
                break

            for name, depth_diff in info["predict_depth"].items():
                units = "(m)"
                color_img_path = os.path.join(save_dir["crop"],  f"{name}.png")
                color_img = cv2.imread(color_img_path)
                save_path_2d = os.path.join(visual_dir, f"{name}_predict_depth_2d.png")
                save_path_3d = os.path.join(visual_dir, f"{name}_predict_depth_3d.png")
                self.visual.plot_2d_deviation(depth_diff, 
                                        color_img, 
                                        cmap='coolwarm', 
                                        save_path=save_path_2d, 
                                        colorbar=f"value{units}",
                                        title=f"2D-Predict-Depth-Visualization")
                self.visual.plot_3d_deviation(depth_diff, 
                                        color_img, 
                                        cmap='coolwarm', 
                                        save_path=save_path_3d,
                                        z="predict-depth-abs",
                                        colorbar=f"value{units}",
                                        title=f"3D-Predict-Depth-Visualization")
                break

        return metrics_result
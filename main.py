import os
import sys
import time
import yaml
import shutil
import cv2
import re
from pathlib import Path
import argparse
from SteroDepthEstimation.utils.define_print import DPrint
from SteroDepthEstimation import stereo_depth_estimation

def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Binocular depth prediction evaluation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--task_type', type=str, default="metrics", 
                        help='task type, opt [gt, calib, metrics, foundation, roi]')
    parser.add_argument('--datasets_root_dir', type=str, default="/mnt/disk1/chao04.chen/horizon/d-robotics/datasets_test", 
                        help='Directory all camera images with timestamp filenames')
    parser.add_argument('--camera_info_path', type=str, default="./calib_fish.json", 
                        help='Camera calibration file path')
    parser.add_argument('--time_window', type=float, default=0.5,
                        help='Time window for searching lidar matches (seconds)')
    parser.add_argument('--time_threshold', type=float, default=0.2,
                        help='Maximum allowed time difference for a valid match (seconds)')
    parser.add_argument('--bin_width', type=int, default=1280,
                        help='Width of bin image')
    parser.add_argument('--bin_height', type=int, default=2176,
                        help='Height of bin image')        
    parser.add_argument('--depth_min', type=float, default=0.0,
                        help='Minimum depth threshold') 
    parser.add_argument('--depth_max', type=float, default=3.0,
                        help='Maximum depth threshold') 
    parser.add_argument('--use_roi_area', type=bool, default=True,
                        help='Whether to use the area of interest')
    parser.add_argument('--use_fixed_depth', type=bool, default=False,
                        help='whether to use fixed depth value')  
    parser.add_argument('--use_history_data', type=bool, default=False,
                        help='whether to use history data')  
    parser.add_argument('--debug', type=bool, default=False,
                        help='whether to turn on debug')     
    parser.add_argument('--use_foundation', type=bool, default=False,
                        help='whether to use Foundation stereo gt ')  
    parser.add_argument('--coord_offest', nargs='+', type=int, default=[15, 15, 15, 15], 
                        help='Coordinate offset value')
    parser.add_argument('--use_category', nargs='+', type=str, default=["box", "carton"], 
                        help='ROI category')
    return parser.parse_args()


def main():
    args = parse_arguments()
    depth_estimation = stereo_depth_estimation.SteroDepthEstimation(args)
    if args.task_type == "gt":
        DPrint.print_color_message("Start Run GT Task", "RED")
        depth_estimation.depth_gt()
        DPrint.print_color_message("End Run GT Task", "RED")
    elif args.task_type == "foundation":
        DPrint.print_color_message("Start Run Foundation Task", "RED")
        depth_estimation.depth_gt_foundation()
        DPrint.print_color_message("End Run Foundation Task", "RED")  
    elif args.task_type == "metrics":
        DPrint.print_color_message("Start Run Metrics Task", "RED")
        depth_estimation.evaluate()
        DPrint.print_color_message("End Run Metrics Task", "RED")
    elif args.task_type == "calib":
        DPrint.print_color_message("Start Run Calib Task", "RED")
        raw_dir = "/mnt/disk1/chao04.chen/horizon/d-robotics/code/Depth_Estimation_new/stereo_calib/calib_datasets/calib_yg_gz_20250624/raw"
        py_path = "./stereo_calib/calib.py"
        os.system(f"python {py_path} --raw_dir {raw_dir} ")
        DPrint.print_color_message("End Run Metrics Task", "RED")
    elif args.task_type == "roi": 
        DPrint.print_color_message("Start Run ROI Task", "RED")
        token = "b705b99ec89dec76e74b0eb91905adb8"
        datasets_root_dir = args.datasets_root_dir
        use_category = args.use_category
        single_detect = True
        py_path = "./dds-cloudapi-sdk/detect.py"
        os.system(f"python {py_path} --token {token}  --datasets_root_dir {datasets_root_dir} --use_category {use_category} --single_detect {single_detect}")
        DPrint.print_color_message("End Run ROI Task", "RED")
    else:
        pass
    

if __name__ == "__main__":
    main()
import os
import shutil
import argparse
import numpy as np
import json
# import colormsg
import time
import sys
from utils.tool import Tools
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict
from utils.define_print import DPrint

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

class SensorSynch:
    def __init__(self, time_window:float, time_threshold:float):
        self.time_window = time_window
        self.time_threshold = time_threshold
        print(f"time_window = {self.time_window:.3f}s | time_threshold = {self.time_threshold:.3f}s")

    def align_sensor_data(self, camera_dir:str, lidar_dir:str, output_dir:str) -> None:
        """
        Aligns camera and LiDAR data based on timestamps and copies matched files to the output directory.

        parameters:
           camera_dir (str): Directory path to the camera data files (.bin).
            lidar_dir (str): Directory path to the LiDAR data files (.pcd).
            output_dir (str): Directory path where matched pairs will be copied.

        Returns:
            None.
        """
        DPrint.print_color_message("Sensor time Synch Start", "BLUE")
        # 目录不存在创建，存在清空目录
        Tools.prepare_directory(output_dir)
        camera_files = [f for f in os.listdir(camera_dir) if f.endswith('.bin')]
        camera_timestamps = {}
        for f in camera_files:
            try:
                ts = int(os.path.splitext(f)[0])
                camera_timestamps[ts] = os.path.join(camera_dir, f)
            except ValueError:
                continue

        lidar_files = [f for f in os.listdir(lidar_dir) if f.endswith('.pcd')]
        lidar_timestamps = {}
        for f in lidar_files:
            try:
                ts = int(os.path.splitext(f)[0])
                lidar_timestamps[ts] = os.path.join(lidar_dir, f)
            except ValueError:
                continue

        if not camera_timestamps or not lidar_timestamps:
            print("error, data is empty !")
            return

        # 将时间戳转换为排序数组以提高搜索效率
        camera_ts_sorted = np.sort(list(camera_timestamps.keys()))
        lidar_ts_sorted = np.sort(list(lidar_timestamps.keys()))

        matched_pairs =[]
        time_diffs =  []
        skipped_cameras = 0
        
        print(f"camera num = {len(camera_ts_sorted)}, lidar num = {len(lidar_ts_sorted)}")

        pbar = tqdm(total=len(camera_ts_sorted))
        for cam_ts in camera_ts_sorted:
            lidar_match = None
            min_diff = float('inf')
            idx = np.searchsorted(lidar_ts_sorted, cam_ts)
            for i in range(max(0, idx-5), min(len(lidar_ts_sorted), idx+5)):
                lidar_ts = lidar_ts_sorted[i]
                time_diff = abs(cam_ts - lidar_ts)
                if time_diff > self.time_window * 1000000000:
                    continue
                if time_diff < min_diff:
                    min_diff = time_diff
                    lidar_match = lidar_ts
            if lidar_match is not None and min_diff <= self.time_threshold * 1000000000:
                cam_src = camera_timestamps[cam_ts]
                cam_dst = os.path.join(output_dir, f"{cam_ts}.bin")
                shutil.copy2(cam_src, cam_dst)
                
                lidar_src = lidar_timestamps[lidar_match]
                lidar_dst = os.path.join(output_dir, f"{cam_ts}.pcd")
                # print(f"lidar_src={lidar_src}, lidar_dst={lidar_dst}")
                shutil.copy2(lidar_src, lidar_dst)
                
                # 记录匹配信息
                matched_pairs.append({
                    'camera_timestamp': cam_ts,
                    'lidar_timestamp': lidar_match,
                    'time_diff': min_diff,
                    'camera_path': cam_dst,
                    'lidar_path': lidar_dst
                })
                
                time_diffs.append(min_diff)
            else:
                skipped_cameras += 1
            pbar.update(1)
        pbar.close()
        
        # 保存匹配结果
        time_diffs = [x / 1000000000 for x in time_diffs]
        stats = {
            'total_camera_frames': len(camera_ts_sorted),
            'total_lidar_frames': len(lidar_ts_sorted),
            'matched_pairs': len(matched_pairs),
            'skipped_cameras': skipped_cameras,
            'match_rate': len(matched_pairs) / len(camera_ts_sorted) * 100,
            'time_window': self.time_window,
            'time_threshold': self.time_threshold,
            'min_time_diff': min(time_diffs) if time_diffs else 0,
            'max_time_diff': max(time_diffs) if time_diffs else 0,
            'avg_time_diff': sum(time_diffs) / len(time_diffs) if time_diffs else 0,
            'median_time_diff': sorted(time_diffs)[len(time_diffs)//2] if time_diffs else 0,
            'matches': matched_pairs
        }
        # 保存统计信息
        # stats_file = os.path.join(output_dir, 'alignment_stats.json')
        # with open(stats_file, 'w') as f:
        #     json.dump(stats, f, indent=2)
        # print(f"统计信息已保存至: {stats_file}")

        # 生成时间差分布图
        # if time_diffs:
        #     plt.figure(figsize=(10, 6))
        #     plt.hist(time_diffs, bins=50, alpha=0.7, color='blue')
        #     plt.axvline(time_threshold, color='red', linestyle='dashed', linewidth=2, label=f'Threshold ({time_threshold:.3f}s)')
        #     plt.xlabel('Time Difference (seconds)')
        #     plt.ylabel('Count')
        #     plt.title('Camera-Lidar Time Synchronization')
        #     plt.legend()
        #     plt.grid(True)
        #     plot_file = os.path.join(output_dir, 'time_difference_distribution.png')
        #     plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        #     plt.close()
        #     if time_diffs:
        #        print(f"时间差分布图已保存至: {plot_file}")
        
        print("\n====== 对齐结果 ======\n")
        print(f"相机帧总数: {stats['total_camera_frames']}")
        print(f"激光雷达帧总数: {stats['total_lidar_frames']}")
        print(f"成功匹配对数: {stats['matched_pairs']}")
        print(f"跳过相机帧: {stats['skipped_cameras']}")
        print(f"匹配率: {stats['match_rate']:.2f}%")
        print(f"最小时间差: {stats['min_time_diff']:.6f}s")
        print(f"最大时间差: {stats['max_time_diff']:.6f}s")
        print(f"平均时间差: {stats['avg_time_diff']:.6f}s")
        print(f"中位数时间差: {stats['median_time_diff']:.6f}s")
        DPrint.print_color_message("Sensor time Synch End", "BLUE")
        

def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Align camera and lidar data based on timestamps',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--camera_dir', required=True, 
                        help='Directory containing camera images with timestamp filenames')
    parser.add_argument('--lidar_dir', required=True, 
                        help='Directory containing lidar point clouds with timestamp filenames')
    parser.add_argument('--output_dir', required=True, 
                        help='Output directory for aligned data')
    parser.add_argument('--time_window', type=float, default=0.1,
                        help='Time window for searching lidar matches (seconds)')
    parser.add_argument('--time_threshold', type=float, default=0.05,
                        help='Maximum allowed time difference for a valid match (seconds)')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    start_time = time.time()
    sensor_syn = SensorSynch(0.5, 0.2)
    sensor_syn.align_sensor_data(
                                camera_dir=args.camera_dir,
                                lidar_dir=args.lidar_dir,
                                output_dir=args.output_dir,
                                time_window=args.time_window,
                                time_threshold=args.time_threshold
                            )
    print(f"process end! time = {time.time()-start_time:.2f}s")

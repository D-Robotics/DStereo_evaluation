import numpy as np
import cv2
import os
import sys
from pathlib import Path
from utils.pointcloud_tool import PointCloudUtils
# from pointcloud_tool import PointCloudUtils
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

class Visual:
    def __init__(self):
        pass
    
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

    def render_depth_difference_as_color(self, 
                                        diff_image:dict, 
                                        visual_img_dir:str, 
                                        color_map:np.array=None, 
                                        min_value:float=0.0, 
                                        max_value:float=5.0, 
                                        suffix:str="diff") -> None:
        """
        Render a single-channel depth difference map as a color image, 
        with areas of greater deviation shown in increasingly intense red

        Parameters:
            diff_image (dict): A dictionary where keys are filenames and values are depth difference images.
            visual_img_dir (str): The directory where the rendered color images will be saved.
            color_map (np.array, optional): A custom color map to apply to the normalized depth difference values. Defaults to None.
            min_value (float, optional): The minimum value for normalization. Defaults to 0.0.
            max_value (float, optional): The maximum value for normalization. Defaults to 5.0.
            suffix (str, optional): The suffix to append to the saved image filenames. Defaults to "diff".

        return:
            None
        """
        self.create_folder(visual_img_dir)
        for filename, image in diff_image.items():
            image_normalized = np.nan_to_num(image, nan=0.0)
            if min_value is None:
                min_value = np.min(image_normalized)
            if max_value is None:
                max_value = np.max(image_normalized)

            # 归一化深度差值图
            normalized = (image_normalized - min_value) / (max_value - min_value)
            # 默认使用红色渐变
            if color_map is None:
                color_image = np.zeros((image_normalized.shape[0], image_normalized.shape[1], 3), dtype=np.uint8)
                # color_image[..., 2] = (normalized_depth * 255).astype(np.uint8) 
                color_image[:, :, 0] = (normalized * 255).astype(np.uint8) 
                color_image[:, :, 1] = 0  
                color_image[:, :, 2] = 0 
            else:
                # 使用自定义颜色映射函数
                color_image = color_map(normalized)
            nan_mask = np.isnan(image)
            color_image[nan_mask, 0] = 0  
            color_image[nan_mask, 1] = 255  
            color_image[nan_mask, 2] = 0 

            name = Path(filename).stem
            image_path = os.path.join(visual_img_dir, f"{name}_{suffix}")
            cv2.imwrite(image_path, color_image)


    def visualize_error_distribution(self, 
                                    errors:dict, 
                                    min_vale, 
                                    max_vale, 
                                    delta, 
                                    title="distribution-interval", 
                                    xlabel="error-interval", 
                                    ylabel="ratio(%)", 
                                    save_name="./visual.png") -> tuple:
        """
        Visualizes the distribution of error values in specified intervals.

        Parameters:
            errors (dict): A dictionary where keys are filenames and values are lists of error values.
            min_value (float): The minimum value for the error intervals.
            max_value (float): The maximum value for the error intervals.
            delta (float): The step size for the intervals.
            title (str, optional): The title of the plot. Defaults to "distribution-interval".
            xlabel (str, optional): The label for the x-axis. Defaults to "error-interval".
            ylabel (str, optional): The label for the y-axis. Defaults to "ratio(%)".
            save_name (str, optional): The file path to save the plot. Defaults to "./visual.png".

        return:
            tuple:
                bins (list): list of  interval 
                counts (list): the number of error values each interval
                percentage (list): ratio 
        """
        # 将误差值转换为 NumPy 数组
        errors = np.array(errors)
        
        # 确定区间范围
        min_error = np.min(errors)
        max_error = np.max(errors)
        bins = np.arange(min_vale, max_vale, delta)
        
        # 统计每个区间的误差值数量
        counts, _ = np.histogram(errors, bins=bins)
        
        # 计算每个区间的占比
        total_count = len(errors)
        percentages = counts / total_count
        
        # 可视化
        plt.figure(figsize=(15, 10))
        plt.bar(range(len(bins) - 1), percentages, tick_label=[f"[{b[0]:.2f}, {b[1]:.2f})" for b in zip(bins[:-1], bins[1:])])
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.xticks(rotation=45)
        plt.tight_layout()
        # plt.show()
        plt.savefig(save_name)
        # 返回区间范围、每个区间的误差值数量和占比
        return list(zip(bins[:-1], bins[1:])), counts, percentages

    def visual_valid_deviation(self, 
                        img_dir:str, 
                        diff_vale:dict, 
                        valid_index:dict, 
                        visual_save_dir:str, 
                        diff_th:float=1, 
                        suffix:str="depth_diff") -> None:
        """
        Visualizes the deviation values on the corresponding color images.


        Parameters:
            img_dir (str): The directory containing the color images.
            diff_vale (dict): A dictionary where keys are filenames and values are arrays of deviation values.
            valid_index (dict): A dictionary where keys are filenames and values are arrays of valid indices.
            visual_save_dir (str): The directory where the visualized images will be saved.
            diff_th (float, optional): The threshold for deviation values. Defaults to 1.
            suffix (str, optional): The suffix to append to the saved image filenames. Defaults to "depth_diff".
            
        Returns:
            None
        """
        self.create_folder(visual_save_dir)
        for filename, value in diff_vale.items():
            # 获取有效值索引
            coord_index = valid_index[filename]
            assert len(value) == len(coord_index), "The valid point data length except!"

            # 获取偏差过大的点mask
            mask = value > diff_th
            coords = coord_index[mask]

            color_img_path = os.path.join(img_dir, filename)
            if os.path.exists(color_img_path):
                color_image = cv2.imread(color_img_path)
                height, width, _ = color_image.shape
                # 可视化偏差过大的点
                visual_img = color_image.copy()
                for coord in list(coords):
                    h, w = coord[0], coord[1]
                    if (0 <= w < width) and (0 <= h < height):
                        # visual_img[h, w] = 0.5 * 255 + (1 - 0.5) * visual_img[h][w]
                        cv2.circle(visual_img, (w, h), radius=1, color=(0, 0, 255), thickness=-1)
                name = Path(filename).stem
                cv2.imwrite(os.path.join(visual_img_dir, f"{name}_{suffix}.png"), visual_img)


    def visual_abs_deviation(self, 
                    img_dir:str, 
                    diff_vale:dict, 
                    visual_save_dir:str, 
                    diff_th:float, 
                    suffix:str="depth_error") -> None:
        """
        Visualizes absolute deviation values on the corresponding color images.

        Parameters:
            img_dir (str): The directory containing the color images.
            diff_vale (dict): A dictionary where keys are filenames and values are arrays of absolute deviation values.
            visual_save_dir (str): The directory where the visualized images will be saved.
            diff_th (float): The threshold for deviation values. Deviations greater than this threshold will be highlighted.
            suffix (str, optional): The suffix to append to the saved image filenames. Defaults to "depth_error".
        
        Returns:
            None
        """
        self.create_folder(visual_save_dir)
        for filename, depth_diff in diff_vale.items():
            mask = (depth_diff > diff_th).astype(np.uint8) * 255
            color_img_path = os.path.join(img_dir, filename)
            if os.path.exists(color_img_path):
                color_image = cv2.imread(color_img_path)
                highlighted_img = color_image.copy()
                highlighted_img[mask == 255] = [0, 0, 255]
                color_image[mask == 255] = (0.7 * color_image[mask == 255] + (1 - 0.7) * highlighted_img[mask == 255])
            name = Path(filename).stem
            cv2.imwrite(os.path.join(visual_img_dir, f"{name}_{suffix}.png"), color_image)


    def plot_3d_deviation(self, deviation, color_img, cmap='coolwarm', save_path="3d", z= "", colorbar="depth_value", title="3D-Visualization") -> None:
        """
        Visualizes a 2D deviation array as a 3D surface plot with a color map.

        Parameters:
            deviation (np.ndarray): A 2D array containing deviation values.
            color_img (np.ndarray): A 2D or 3D array representing the color image.
            cmap (str, optional): The colormap to use for the deviation values. Defaults to 'coolwarm'.
            save_path (str, optional): The file path to save the plot. Defaults to "3d.png".
            z (str, optional): The label for the z-axis. Defaults to "Depth Value".
            colorbar (str, optional): The label for the color bar. Defaults to "depth_value".
            title (str, optional): The title of the plot. Defaults to "3D-Visualization".

        Returns:
            None
        """
        if len(deviation.shape) != 2:
            raise ValueError("Input array must be 2D (single-channel)")
        # Create a meshgrid for x and y coordinates
        x = np.arange(deviation.shape[1])
        y = np.arange(deviation.shape[0])
        x, y = np.meshgrid(x, y)

        # Create a figure and a 3D axis
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        color_img = color_img.astype(float) / 255.0
        ax.plot_surface(x, y, 
                        np.zeros_like(deviation),
                        facecolors=color_img,
                        rstride=1, 
                        cstride=1,  # Full resolution
                        shade=False,
                        alpha=1.0,
                        zorder=1)
                
        # ax.invert_xaxis()
        ax.invert_yaxis()

        surf = ax.plot_surface(x, y, 
                            deviation, 
                            cmap=cmap,
                            rstride=1, 
                            cstride=1,
                            linewidth=0.1,
                            antialiased=False,
                            alpha=0.5,
                            zorder=2)

        # Add a color bar
        cbar = fig.colorbar(surf, ax=ax, shrink=1, aspect=10)
        cbar.set_label(colorbar, color='white')
        cbar.ax.yaxis.set_tick_params(color='white')
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')

        # Set labels and title
        ax.set_xlabel('X', color='white')
        ax.set_ylabel('Y', color='white')
        ax.set_zlabel(z, color='white')
        ax.set_title(title, color='white')

        plt.savefig(save_path, dpi=300, transparent=True)
        plt.close()


    def plot_2d_deviation(self, deviation, color_img, cmap='coolwarm', save_path="2d", z= "", colorbar="depth_value", title="2D-Visualization") -> None:
        """
        Visualizes a 2D deviation array as a 2D surface plot with a color map.
        
        Parameters:
            deviation (np.ndarray): A 2D array containing deviation values.
            color_img (np.ndarray): A 2D or 3D array representing the color image.
            cmap (str, optional): The colormap to use for the deviation values. Defaults to 'coolwarm'.
            save_path (str, optional): The file path to save the plot. Defaults to "2d.png".
            colorbar (str, optional): The label for the color bar. Defaults to "depth_value".
            title (str, optional): The title of the plot. Defaults to "2D-Visualization".

        Returns:
            None
        """
        if len(deviation.shape) != 2:
            raise ValueError("Input array must be 2D (single-channel)")
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(color_img, cmap='gray')

        im = ax.imshow(deviation, cmap=cmap, alpha=0.5)

        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(colorbar, color='white')
        cbar.ax.yaxis.set_tick_params(color='white')
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
        
        ax.set_title(title, color='white')
        ax.set_xlabel('X', color='white')
        ax.set_ylabel('Y', color='white')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, transparent=True)
        plt.close()

    def plot_and_save_curve(self, x_values: list, y_values: list, title: str = "Curve Plot", xlabel: str = "X-axis", ylabel: str = "Y-axis", output_path: str = "curve_plot.png") -> None:
        """
        Generates a curve plot from the given x and y values and saves it to the specified output path.

        Parameters:
            x_values (list): A list of values for the x-axis.
            y_values (list): A list of values for the y-axis.
            title (str, optional): Title of the plot. Defaults to "Curve Plot".
            xlabel (str, optional): Label for the x-axis. Defaults to "X-axis".
            ylabel (str, optional): Label for the y-axis. Defaults to "Y-axis".
            output_path (str, optional): Path where the plot will be saved. Defaults to "curve_plot.png".

        Returns:
            None
        """
        # Check if the lengths of the x and y lists are the same
        if len(x_values) != len(y_values):
            raise ValueError("The lengths of x_values and y_values must be the same.")

        # Create the plot
        plt.figure(figsize=(10, 6))
        plt.plot(x_values, y_values, marker='o', linestyle='-')
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(True)

        # Save the plot to the specified output path
        plt.savefig(output_path)
        print(f"Plot saved to {output_path}")

        # Close the plot to free up memory
        plt.close()

if __name__ == "__main__":
    gray_image = np.random.randint(0, 21, (250, 250), dtype=np.uint8)
    visual = Visual()

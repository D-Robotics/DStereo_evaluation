
import cv2
import os
import sys
import json
import numpy as np
from pathlib import Path

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

class ImageTool:
    """
    A comprehensive utility class for image processing.
    """
    def __init__(self):
        pass

    @classmethod
    def crop_image_by_center(cls, image:np.ndarray, center_x:int, center_y:int, crop_width:int, crop_height:int) -> tuple:
        """
        Crops a region from an image based on the specified center coordinates and dimensions.

        Parameters:
            image (np.ndarray): The input image.
            center_x (int): The x-coordinate of the center of the region to crop.
            center_y (int): The y-coordinate of the center of the region to crop.
            crop_width (int): The width of the region to crop.
            crop_height (int): The height of the region to crop.

        Returns:
            tuple: crop status and the cropped image region.
        """
        if (image.shape[0] < crop_height) or (image.shape[1] < crop_width):
            print(f"warning:crop size except!")
            return (False, image)

        # Calculate the boundaries of the region to crop
        start_x = max(0, center_x - crop_width // 2)
        end_x = min(image.shape[1], center_x + crop_width // 2)
        start_y = max(0, center_y - crop_height // 2)
        end_y = min(image.shape[0], center_y + crop_height // 2)

        # Extract the region from the image
        cropped_image = image[start_y:end_y, start_x:end_x]

        # If the cropped region is smaller than the specified dimensions due to image boundaries, pad it with zeros
        if cropped_image.shape[0] < crop_height or cropped_image.shape[1] < crop_width:
            padded_image = np.zeros((crop_height, crop_width, image.shape[2]), dtype=image.dtype)
            pad_top = max(0, crop_height // 2 - center_y)
            pad_bottom = max(0, center_y + crop_height // 2 - image.shape[0])
            pad_left = max(0, crop_width // 2 - center_x)
            pad_right = max(0, center_x + crop_width // 2 - image.shape[1])
            padded_image[pad_top:pad_top + cropped_image.shape[0], pad_left:pad_left + cropped_image.shape[1]] = cropped_image
            return True, padded_image
        else:
            return True, cropped_image

    @classmethod
    def extract_center_region(cls, depth_map:np.ndarray, width:int, height:int) -> tuple:
        """
        Extracts a region from a depth map around the center point with specified width and height.

        Parameters:
            depth_map (np.ndarray): The input depth map as a numpy array.
            width (int): The width of the region to extract.
            height (int): The height of the region to extract.

        Returns:
            tuple: A tuple containing the top-left and bottom-right coordinates of the extracted region.
        """
        # Calculate the center coordinates of the depth map
        center_y, center_x = depth_map.shape[0] // 2, depth_map.shape[1] // 2
        
        # Calculate the top-left and bottom-right coordinates
        top_left = (center_x - width // 2, center_y - height // 2)
        bottom_right = (center_x + width // 2, center_y + height // 2)
        
        # Ensure the coordinates are within the bounds of the depth map
        top_left = (max(0, top_left[0]), max(0, top_left[1]))
        bottom_right = (min(depth_map.shape[1], bottom_right[0]), min(depth_map.shape[0], bottom_right[1]))
        
        return top_left, bottom_right

    @classmethod
    def generate_fixed_depth_map(cls, width:int, height:int, distance_mm:int) -> np.ndarray:
        """
        Generates a depth map with a specified resolution where all depth values are set to a fixed distance in millimeters.

        Parameters:
            width (int): The width of the depth map in pixels.
            height (int): The height of the depth map in pixels.
            distance_mm (int): The fixed depth value in millimeters.

        Returns:
            np.ndarray: A depth map of shape (height, width) with data type np.uint16.
        """
        depth_map = np.full((height, width), distance_mm, dtype=np.uint16)
        return depth_map

    @classmethod
    def extract_detection_box(cls, image_path:str, x_min:int, y_min:int, x_max:int, y_max:int) -> np.ndarray:
        """
        Extracts a region of interest from an image based on detection box coordinates.

        Parameters:
            image_path (str): The path to the input image.
            x_min (int): The minimum x-coordinate of the detection box.
            y_min (int): The minimum y-coordinate of the detection box.
            x_max (int): The maximum x-coordinate of the detection box.
            y_max (int): The maximum y-coordinate of the detection box.

        Returns:
            np.ndarray: The extracted region of interest as an image.
        """
        # Load the image
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Could not load image at path: {image_path}")
        
        # Ensure the detection box coordinates are within the image boundaries
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(image.shape[1], x_max)
        y_max = min(image.shape[0], y_max)
        
        # Extract the detection box region
        detection_box_region = image[y_min:y_max, x_min:x_max]
        
        return detection_box_region

    @classmethod
    def extract_and_save_regions(cls, image_dir:str, depth_dir:str, json_dir:str, output_image_dir:str, output_depth_dir:str) -> None:
        """
        Extracts regions of interest from images and depth maps based on bounding box coordinates
        stored in JSON files, and saves the extracted regions into separate folders.

        Parameters:
            image_dir (str): Directory path to the input images.
            depth_dir (str): Directory path to the input depth maps.
            json_dir (str): Directory path to the JSON files containing bounding box information.
            output_image_dir (str): Directory path to save the extracted image regions.
            output_depth_dir (str): Directory path to save the extracted depth regions.

        Returns:
            None
        """
        # Ensure output directories exist
        os.makedirs(output_image_dir, exist_ok=True)
        os.makedirs(output_depth_dir, exist_ok=True)

        # Iterate over all JSON files in the JSON directory
        for json_file in os.listdir(json_dir):
            if json_file.endswith('.json'):
                # Construct file paths
                image_name = Path(json_file).stem
                json_path = os.path.join(json_dir, json_file)
                image_path = os.path.join(image_dir, json_file.replace('.json', '.png'))
                depth_path = os.path.join(depth_dir, json_file.replace('.json', '.png'))

                # Check if corresponding image and depth files exist
                if not os.path.exists(image_path) or not os.path.exists(depth_path):
                    print(f"Skipping {json_file} due to missing image or depth file.")
                    continue

                # Load JSON file
                with open(json_path, 'r') as f:
                    json_data = json.load(f)

                # Extract bounding box coordinates from JSON
                if 'bbox' in json_data:
                    bbox = json_data["objects"]['bbox']
                    bbox = [int(coord) for coord in bbox]
                    x_min, y_min, x_max, y_max = bbox
                else:
                    print(f"Skipping {json_file} due to missing 'bbox' key in JSON.")
                    continue

                # Load image and depth map
                image = cv2.imread(image_path)
                depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

                if image is None or depth is None:
                    print(f"Skipping {json_file} due to unable to load image or depth.")
                    continue

                if (image.shape[0] != depth.shape[0]) or (image.shape[0] != depth.shape[0]):
                    print(f"Skipping {image_name}.png depth and image shape except.")
                    continue

                # Ensure bounding box coordinates are within image boundaries
                x_min = max(0, x_min)
                y_min = max(0, y_min)
                x_max = min(image.shape[1], x_max)
                y_max = min(image.shape[0], y_max)

                # Extract regions of interest
                image_region = image[y_min:y_max, x_min:x_max]
                depth_region = depth[y_min:y_max, x_min:x_max]

                # Save extracted regions
                output_image_path = os.path.join(output_image_dir, json_file.replace('.json', '.png'))
                output_depth_path = os.path.join(output_depth_dir, json_file.replace('.json', '.png'))
                cv2.imwrite(output_image_path, image_region)
                cv2.imwrite(output_depth_path, depth_region)

                print(f"Saved image region to {output_image_path}, Saved depth region to {output_depth_path}")

    @classmethod
    def visualize_threshold(cls, image:np.ndarray, threshold:float, cmap:str) -> None:
        """
        Visualizes values above a certain threshold in a numpy image using a specified colormap.

        Parameters:
            image (np.ndarray): The input numpy image array.
            threshold (float): The threshold value. Values above this will be visualized.
            cmap (str, optional): The colormap to use for visualization. Defaults to 'jet'.

        Returns:
            None
        """
        # Create a mask for values above the threshold
        mask = image > threshold

        # Create a figure with two subplots
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))

        # Display the original image
        axs[0].imshow(image, cmap='gray')
        axs[0].set_title('Original Image')
        axs[0].axis('off')

        # Display the thresholded image with colormap
        thresholded_image = np.ma.masked_where(~mask, image)
        img = axs[1].imshow(thresholded_image, cmap=cmap)
        axs[1].set_title('Thresholded Image with Colormap')
        axs[1].axis('off')

        # Add a colorbar
        fig.colorbar(img, ax=axs[1], label='Value')

        plt.tight_layout()
        plt.show()
        
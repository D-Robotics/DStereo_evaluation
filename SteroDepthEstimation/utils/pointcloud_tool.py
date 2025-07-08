import numpy as np
import open3d as o3d
import os
import cv2
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)


class PointCloudUtils:
    """
    A comprehensive utility class for point cloud processing.
    """

    def __init__(self):
        pass

    @classmethod
    def read_point_cloud(cls, file_path:str) -> o3d.geometry.PointCloud:
        """
        Reads a point cloud from a PLY or PCD file.

        Parameters:
            file_path (str): Path to the PLY or PCD file.

        Returns:
            o3d.geometry.PointCloud: The loaded point cloud.

        Raises:
            ValueError: If the file format is not supported.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file {file_path} does not exist.")
        if file_path.endswith(".ply") or file_path.endswith(".pcd"):
            point_cloud = o3d.io.read_point_cloud(file_path)
            return point_cloud
        else:
            raise ValueError("Unsupported file format. Only PLY and PCD are supported.")

    @classmethod
    def write_point_cloud(cls, xyz_point:np.ndarray, file_path:str, rgb_img=None) -> None:
        """
        Saves point cloud data to a PLY or PCD file using Open3D.

        Parameters:
            xyz_point (np.ndarray): Nx3 array of point coordinates.
            file_path (str): Path to save the PLY or PCD file.
            rgb_img (np.ndarray): Nx3 array of point colors in RGB format.

        Returns:
            None

        Raises:
            ValueError: If the file format is not supported.
        """

        # Check if the file format is supported
        if not file_path.endswith((".ply", ".pcd")):
            raise ValueError("Unsupported file format. Only PLY and PCD are supported.")

        # Create an Open3D point cloud object
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz_point)

        if not rgb_img:
            pixel_num = rgb_img.shape[0] * rgb_img.shape[1]
            if xyz_point.shape[0] != pixel_num:
                raise ValueError(
                    "The number of points in xyz_points and rgb arrays must match."
                )
            # Set points and colors
            pcd.colors = o3d.utility.Vector3dVector(
                rgb_img / 255.0
            )  # Normalize colors to [0, 1]

        # Save the point cloud
        o3d.io.write_point_cloud(file_path, pcd)

    @classmethod
    def visualize_point_cloud(cls, point_cloud) -> None:
        """
        Visualizes the point cloud using Open3D's visualization tools.

        Parameters:
            point_cloud (o3d.geometry.PointCloud): The visual point cloud.

        Returns:
            None
        """
        o3d.visualization.draw_geometries([point_cloud])

    @classmethod
    def get_point_cloud_info(cls, point_cloud:o3d.geometry.PointCloud) -> dict:
        """
        Retrieves basic information about the point cloud.

        Parameters:
            point_cloud (o3d.geometry.PointCloud): input point cloud.

        Returns:
            dict: A dictionary containing information about the point cloud.
        """
        return {
            "number_of_points": len(point_cloud.points),
            "has_colors": point_cloud.colors is not None,
            "has_normals": point_cloud.normals is not None,
        }

    @classmethod
    def ply_to_pcd(cls, input_path:str, output_path:str) -> None:
        """
        Converts a PLY file to a PCD file.

        Parameters:
            input_path (str): Path to the input PLY file.
            output_path (str): Path to save the output PCD file.

        Returns:
            None
        """
        point_cloud = self.read_point_cloud(input_path)
        self.write_point_cloud(output_path, point_cloud)

    @classmethod
    def create_transformation_matrix(cls, rotation_matrix:np.ndarray, translation_vector:np.ndarray) -> np.ndarray:
        """
        Creates a 4x4 transformation matrix from a rotation matrix and a translation vector.

        Parameters:
            rotation_matrix (np.ndarray): 3x3 rotation matrix.
            translation_vector (np.ndarray): 3x1 translation vector.

        Returns:
            np.ndarray: 4x4 transformation matrix.

        Raises:
            ValueError: If the rotation matrix is not 3x3 or the translation vector is not 3x1.
        """
        # Check if the rotation matrix is 3x3
        if rotation_matrix.shape != (3, 3):
            raise ValueError("Rotation matrix must be 3x3.")

        # Check if the translation vector is 3x1
        if translation_vector.shape != (3,):
            raise ValueError("Translation vector must be 3x1.")

        # Create the transformation matrix
        transformation_matrix = np.eye(4)
        transformation_matrix[:3, :3] = rotation_matrix
        transformation_matrix[:3, 3] = translation_vector

        return transformation_matrix

    @classmethod
    def transform_coord_system(cls, points_3d:np.ndarray, transformation_matrix:np.ndarray) -> np.ndarray:
        """
        Transforms the point cloud from one coordinate system to another using a 4x4 transformation matrix.

        Parameters:
            points_3d (np.ndarray): The input point cloud 
            transformation_matrix (np.ndarray): 4x4 transformation matrix

        Returns:
            point cloud (np.ndarray)
        """
        points_homogeneous = np.concatenate([points_3d, np.ones((points_3d.shape[0], 1))], axis=1)
        points_homogeneous = transformation_matrix @ points_homogeneous.T
        points = points_homogeneous[:3, :].T
        return points

    @classmethod
    def transform_point_cloud(cls, point_cloud:o3d.geometry.PointCloud, transformation_matrix:np.ndarray) -> o3d.geometry.PointCloud:
        """
        Transforms the point cloud from one coordinate system to another using a 4x4 transformation matrix.

        Parameters:
            point_cloud (o3d.geometry.PointCloud): The input point cloud.
            transformation_matrix (np.ndarray): 4x4 transformation matrix.

        Returns:
            o3d.geometry.PointCloud: The transformed point cloud.

        Raises:
            ValueError: If the transformation matrix is not 4x4.
        """
        # Check if the transformation matrix is 4x4
        if transformation_matrix.shape != (4, 4):
            raise ValueError("Transformation matrix must be 4x4.")

        # Make a copy of the input point cloud to avoid modifying it in place
        transformed_cloud = o3d.geometry.PointCloud(point_cloud)

        # Apply the transformation
        transformed_cloud.transform(transformation_matrix)

        return transformed_cloud

    @classmethod
    def filter_by_intensity(cls, point_cloud:o3d.geometry.PointCloud, intensity_threshold:int, index:int=0) -> o3d.geometry.PointCloud:
        """
        Filters the point cloud based on intensity values.

        Parameters:
            point_cloud (o3d.geometry.PointCloud): The input point cloud.
            intensity_threshold (float): The minimum intensity value for a point to be retained.
            index (int, optional): intensity is stored in the index channel. Defaults to 0.

        Returns:
            o3d.geometry.PointCloud: The filtered point cloud.
        """
        points = np.asarray(point_cloud.points)
        intensities = np.asarray(point_cloud.colors)[:, index]
        mask = intensities >= intensity_threshold
        filtered_points = o3d.utility.Vector3dVector(points[mask])
        filtered_colors = o3d.utility.Vector3dVector(point_cloud.colors[mask])
        filtered_cloud = o3d.geometry.PointCloud()
        filtered_cloud.points = filtered_points
        filtered_cloud.colors = filtered_colors
        return filtered_cloud


    @classmethod
    def point_cloud_to_depth_map(
        cls, points:np.ndarray, width:int, height:int, fx:float, fy:float, cx:float, cy:float, scale:int=1000, min_depth:float=0.0, max_depth:float=5.0
    ) -> tuple:
        """
        Converts the point cloud to a depth map.

        Parameters:
            points (np.ndarray): Nx3 array of point coordinates.
            width (int): Width of the depth map.
            height (int): Height of the depth map.
            fx (float): Focal length in the x direction.
            fy (float): Focal length in the y direction.
            cx (float): Optical center x coordinate.
            cy (float): Optical center y coordinate.
            scale (int, optional): Scale factor of depth value. Defaults to 1000.
            min_depth (float, optional): Minimum depth value. Defaults to None.
            max_depth (float, optional): Maximum depth value. Defaults to None.

        Returns:
            tuple:
                A 2D depth map array and a list of valid point coordinates.
        """
        # Initialize depth map
        depth_map = np.zeros((height, width), dtype=np.uint16)
        save_valid_points = []
        for point in points:
            x, y, z = point
            if (z > min_depth) and (z <= max_depth):
                u = int(cx + (x * fx) / z)
                v = int(cy + (y * fy) / z)
                if (0 <= u < width) and (0 <= v < height):
                    depth_map[v, u] = int(z * scale)
                    save_valid_points.append((u, v))
        return depth_map, save_valid_points


    @classmethod
    def depth_map_to_point_cloud(
        cls, depth_map:np.ndarray, fx:float, cx:float, fy:float, cy:float, scale:int=1000, color_map:np.ndarray=None) -> o3d.geometry.PointCloud:
        """
        Converts a depth map to a point cloud with optional color information.

        Parameters:
            depth_map (np.ndarray): 2D depth map array.
            fx (float): Focal length in the x direction.
            cx (float): Optical center x coordinate.
            fy (float): Focal length in the y direction.
            cy (float): Optical center y coordinate.
            scale (int, optional): Scale factor for depth values. Defaults to 1000.
            color_map (np.ndarray, optional): Color map array of shape (height, width, 3) with values in [0, 255]. Defaults to None.

        Returns:
            o3d.geometry.PointCloud: The generated point cloud.
        """
        point_cloud = o3d.geometry.PointCloud()
        height, width = depth_map.shape
        if color_map is not None:
            color_height, color_width, _ = color_map.shape
            if (color_height != height) or (color_width != width):
                raise ValueError("The depth_map size and color_map size must match.")
        points = []
        colors = []
        for y in range(height):
            for x in range(width):
                depth = depth_map[y, x]
                if depth == 0:
                    continue
                z = depth / scale 
                x_coord = (x - cx) * z / fx
                y_coord = (y - cy) * z / fy
                points.append([x_coord, y_coord, z])
                bgr = color_map[y, x]
                colors.append([bgr[2]/255.0, bgr[1]/255.0, bgr[0]/255.0])

        if len(points):
            points = np.array(points)
            colors = np.array(colors)
            point_cloud.points = o3d.utility.Vector3dVector(points)
            point_cloud.colors = o3d.utility.Vector3dVector(colors)
        return point_cloud

    @classmethod
    def downsample_point_cloud(cls, point_cloud:o3d.geometry.PointCloud, voxel_size:float) -> o3d.geometry.PointCloud:
        """
        Downsamples the point cloud using voxel grid filtering.

        Parameters:
            point_cloud (o3d.geometry.PointCloud): The input point cloud.
            voxel_size (float): Size of the voxel grid.

        Returns:
            o3d.geometry.PointCloud: The generated sample point cloud.
        """
        point_cloud = point_cloud.voxel_down_sample(voxel_size)
        return point_cloud

    @classmethod
    def estimate_normals(cls, point_cloud:o3d.geometry.PointCloud, radius:float, max_nn:int) -> o3d.geometry.PointCloud:
        """
        Estimates normals for the point cloud.

        Parameters:
            point_cloud (o3d.geometry.PointCloud): The input point cloud.
            radius (float): Radius for normal estimation.
            max_nn (int): Maximum number of nearest neighbors.

        Returns:
            o3d.geometry.PointCloud: The point cloud with estimated normals.

        Raises:
            ValueError: If the point cloud has no points.
        """
        # Check if the point cloud has points
        if len(point_cloud.points) == 0:
            raise ValueError("The point cloud has no points.")

        # Make a copy of the input point cloud to avoid modifying it in place
        point_cloud_with_normals = o3d.geometry.PointCloud(point_cloud)

        # Estimate normals
        point_cloud_with_normals.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=radius, max_nn=max_nn
            )
        )

        # Normalize the normals to ensure they are unit vectors
        normals = np.asarray(point_cloud_with_normals.normals)
        normals /= np.linalg.norm(normals, axis=1, keepdims=True)
        point_cloud_with_normals.normals = o3d.utility.Vector3dVector(normals)

        return point_cloud_with_normals

    @classmethod
    def remove_outliers(cls, point_cloud:o3d.geometry.PointCloud, nb_points:int, radius:float) -> o3d.geometry.PointCloud:
        """
        Removes outlier points from the point cloud based on the number of neighboring points within a specified radius.

        Parameters:
            point_cloud (o3d.geometry.PointCloud): The input point cloud.
            nb_points (int): Minimum number of points required in the neighborhood for a point to be considered valid.
            radius (float): Radius for the neighborhood search.

        Returns:
            o3d.geometry.PointCloud: The point cloud with outliers removed.

        Raises:
            ValueError: If the point cloud has no points or if parameters are invalid.
        """
        # Check if the point cloud has points
        if len(point_cloud.points) == 0:
            raise ValueError("The point cloud has no points.")
        # Check for valid parameters
        if nb_points <= 0:
            raise ValueError("nb_points must be greater than 0.")
        if radius <= 0:
            raise ValueError("radius must be greater than 0.")

        # Create a copy of the input point cloud to avoid modifying it in place
        point_cloud_filtered = o3d.geometry.PointCloud(point_cloud)

        # Remove outliers
        _, indices = point_cloud_filtered.remove_radius_outlier(
            nb_points=nb_points, radius=radius
        )

        # Select points by index
        point_cloud_filtered = point_cloud_filtered.select_by_index(indices)

        return point_cloud_filtered

    @classmethod
    def fit_plane(cls, point_cloud:o3d.geometry.PointCloud, distance_threshold:float, ransac_n:int, num_iterations:int) -> tuple:
        """
        Fits a plane to the point cloud using RANSAC.

        Parameters:
            point_cloud (o3d.geometry.PointCloud): The input point cloud.
            distance_threshold (float): Maximum distance a point can be from the plane to be considered an inlier.
            ransac_n (int): Number of points to sample for RANSAC.
            num_iterations (int): Number of RANSAC iterations.

        Returns:
            tuple: 
                A tuple containing the plane model coefficients (a, b, c, d) and the inlier points.
                The plane equation is ax + by + cz + d = 0.
            None: If no plane is found.

        Raises:
            ValueError: If the point cloud has no points or if parameters are invalid.
        """
        # Check if the point cloud has points
        if len(point_cloud.points) == 0:
            raise ValueError("The point cloud has no points.")
        # Check for valid parameters
        if distance_threshold <= 0:
            raise ValueError("distance_threshold must be greater than 0.")
        if ransac_n <= 0:
            raise ValueError("ransac_n must be greater than 0.")
        if num_iterations <= 0:
            raise ValueError("num_iterations must be greater than 0.")

        # Make a copy of the input point cloud to avoid modifying it in place
        point_cloud_copy = o3d.geometry.PointCloud(point_cloud)

        # Estimate normals for better plane fitting
        point_cloud_copy.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
        )

        # Fit a plane using RANSAC
        plane_model, inliers = point_cloud_copy.segment_plane(
            distance_threshold=distance_threshold,
            ransac_n=ransac_n,
            num_iterations=num_iterations,
        )

        # Extract inlier points
        inlier_points = point_cloud_copy.select_by_index(inliers)

        # Normalize the plane equation to ensure the normal vector is a unit vector
        a, b, c, d = plane_model
        norm = np.sqrt(a**2 + b**2 + c**2)
        if norm != 0:
            a, b, c, d = a / norm, b / norm, c / norm, d / norm

        return (a, b, c, d), inlier_points

import json
import cv2
import os
import numpy as np

class CamIntrinsics:
    def __init__(self, fx:float, fy:float, cx:float, cy:float, scale:int, width:int, height:int):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.scale = scale
        self.width = width
        self.height = height

class CameraParams:
    def __init__(self):
        self.intrinsic = None
        self.extrinsic = None

    def set_intrinsic(self, fx: float, fy: float, cx: float, cy: float) -> None:
        """
        Sets the intrinsic parameters of the camera.

        Parameters:
            fx (float): Focal length in the x direction.
            fy (float): Focal length in the y direction.
            cx (float): Optical center x coordinate.
            cy (float): Optical center y coordinate.

        Returns:
            None
        """
        self.intrinsic = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    def set_extrinsic(self, rotation_matrix: np.ndarray, translation_vector: np.ndarray) -> None:
        """
        Sets the extrinsic parameters of the camera.

        Parameters:
            rotation_matrix (np.ndarray): 3x3 rotation matrix.
            translation_vector (np.ndarray): 3x1 translation vector.
            
        """
        self.extrinsic = np.eye(4)
        self.extrinsic[:3, :3] = rotation_matrix
        self.extrinsic[:3, 3] = translation_vector

    def get_intrinsic(self) -> np.ndarray:
        """
        Retrieves the intrinsic parameters of the camera.

        Returns:
            np.ndarray: 3x3 intrinsic matrix.
        """
        if self.intrinsic is None:
            raise ValueError("Intrinsic parameters are not set.")
        return self.intrinsic

    def get_extrinsic(self) -> np.ndarray:
        """
        Retrieves the extrinsic parameters of the camera.

        Returns:
            np.ndarray: 4x4 extrinsic matrix.
        """
        if self.extrinsic is None:
            raise ValueError("Extrinsic parameters are not set.")
        return self.extrinsic

    def compute_new_intrinsic(self, original_width: int, original_height: int, new_width: int, new_height: int) -> np.ndarray:
        """
        Computes new intrinsic parameters based on the scaling of image dimensions.

        Parameters:
            original_width (int): Original width of the image.
            original_height (int): Original height of the image.
            new_width (int): New width of the image.
            new_height (int): New height of the image.

        Returns:
            np.ndarray: New 3x3 intrinsic matrix adjusted for the image dimension scaling.
        """
        if self.intrinsic is None:
            raise ValueError("Intrinsic parameters are not set.")

        width_scale = new_width / original_width
        height_scale = new_height / original_height

        new_fx = self.intrinsic[0, 0] * width_scale
        new_fy = self.intrinsic[1, 1] * height_scale
        new_cx = self.intrinsic[0, 2] * width_scale
        new_cy = self.intrinsic[1, 2] * height_scale

        new_intrinsic = np.array([[new_fx, 0, new_cx], [0, new_fy, new_cy], [0, 0, 1]])

        return new_intrinsic

class SteroCalibInfo:
    """
    Parameters of the binocluer camera 
    """
    def __init__(self, path_json:str, model:str="fish", camera_to_base:float=3.5):
        self.F = None
        self.B = None
        self.width_l = None
        self.height_l = None
        self.width_r = None
        self.height_r = None
        self.intr_l = None
        self.distort_l = None
        self.intr_r = None
        self.distort_r = None
        self.R = None
        self.t = None
        self.map1_l = None
        self.map2_l = None
        self.map1_r = None
        self.map2_r = None
        self.model = model
        self.LindarToCam = None
        self.load_json(path_json)
        self.cameToBase = camera_to_base

    # 加载相机参数
    def load_json(self, path_json:str) -> None:
        """
        Loads camera calibration parameters from a JSON file and initializes the object's attributes.

        Parameters:
            path_json (str): The file path to the JSON configuration file containing camera calibration parameters.

        Returns:
            None
        """
        with open(path_json, 'r', encoding='utf-8') as f:
            calib = json.load(f)

        self.F = calib["Stereo"]["F"]
        self.B = calib["Stereo"]["B"]
        self.intr = np.array([
            [calib['Stereo']['F'], 0, calib['Stereo']['Cx']],
            [0, calib['Stereo']['F'], calib['Stereo']['Cy']],
            [0, 0, 1]
        ])
        
        self.use_width = calib['Stereo']['use_Width']
        self.use_height = calib['Stereo']['use_Height']

        self.width_l, self.height_l = calib['Left']['Width'], calib['Left']['Height']
        self.width_r, self.height_r = calib['Right']['Width'], calib['Right']['Height']

        self.intr_l = np.array([
            [calib['Left']['Fx'], 0, calib['Left']['Cx']],
            [0, calib['Left']['Fy'], calib['Left']['Cy']],
            [0, 0, 1]
        ])
    
        self.intr_r = np.array([
            [calib['Right']['Fx'], 0, calib['Right']['Cx']],
            [0, calib['Right']['Fy'], calib['Right']['Cy']],
            [0, 0, 1]
        ])
        if self.model == "fish":
            self.distort_l = np.array([[calib['Left']['θ1'], calib['Left']['θ2'],
                                        calib['Left']['θ3'], calib['Left']['θ4']
                                        ]])

            self.distort_r = np.array([[calib['Right']['θ1'], calib['Right']['θ2'],
                                        calib['Right']['θ3'], calib['Right']['θ4']
                                        ]])  
        else:
            self.distort_l = np.array([[calib['Left']['K1'], calib['Left']['K2'],
                                        calib['Left']['P1'], calib['Left']['P2'], calib['Left']['K3'],
                                        calib['Left']['K4'], calib['Left']['K5'], calib['Left']['K6']
                                        ]])

            self.distort_r = np.array([[calib['Right']['K1'], calib['Right']['K2'],
                                        calib['Right']['P1'], calib['Right']['P2'], calib['Right']['K3'],
                                        calib['Right']['K4'], calib['Right']['K5'], calib['Right']['K6']
                                        ]])
        # 旋转矩阵和平移向量
        self.R = np.array(calib['Rotate'])
        self.t = np.array(calib['Trans'])
        self.LindarToCam = np.array(calib['LidarToCam'])
        print(f'****** Load [{path_json}] Success******')

    # 计算标定映射表
    def calc_map(self, rect_w:int=-1, rect_h:int=-1, fov_scale:float=0.75) -> None:
        """
        Calculates rectification maps for stereo image rectification based on camera parameters.

        Parameters:
            rect_w (int, optional): The desired width of the rectified images. Defaults to -1, which means the original image width is used.
            rect_h (int, optional): The desired height of the rectified images. Defaults to -1, which means the original image height is used.
            fov_scale (float, optional): Scaling factor for the field of view, used primarily for fisheye camera models. Defaults to 0.75.

        Returns:
            None
        """
        if self.intr_l is None:
            raise ValueError('Internal and external parameters are not calculated!')

        # 计算相机的修正变换
        size = (self.width_l, self.height_l)
        if rect_w != -1 and rect_h != -1:
            new_size = (rect_w, rect_h)
        else:
            new_size = size
        if self.model == 'pinhole':
            flags = cv2.CALIB_ZERO_DISPARITY  # 主点一致
            # flags = 0  # 主点不一致
            # alpha = -1
            alpha = 0
            # alpha = 0.5
            # alpha = 1
            R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(self.intr_l, self.distort_l,
                                                        self.intr_r, self.distort_r,
                                                        size, self.R, self.t, flags=flags, alpha=alpha,
                                                        newImageSize=new_size)
            self.Focal, self.B, self.cx, self.cy, self.doffs = Q[2, 3], 1 / Q[3, 2], -Q[0, 3], -Q[1, 3], Q[3, 3] / Q[
                3, 2]

            # 计算映射矩阵LUT，CV_32FC2时map2为空，CV_16SC2时map2为提升定点精度的查找表
            self.map1_l, self.map2_l = cv2.initUndistortRectifyMap(self.intr_l, self.distort_l, R1, P1, new_size,
                                                                   cv2.CV_32FC2)
            self.map1_r, self.map2_r = cv2.initUndistortRectifyMap(self.intr_r, self.distort_r, R2, P2, new_size,
                                                                   cv2.CV_32FC2)
        elif self.model == 'fish':
            # flags = cv2.fisheye.CALIB_ZERO_DISPARITY
            balance = 0
            fov_scale = 0.73
            R1, R2, P1, P2, Q = cv2.fisheye.stereoRectify(self.intr_l, self.distort_l, self.intr_r, self.distort_r,
                                                          size, self.R, self.t, flags=cv2.fisheye.CALIB_ZERO_DISPARITY, 
                                                          newImageSize=new_size,
                                                          balance=balance, fov_scale=fov_scale)
            # print(f"R1 = {R1}")
            self.Focal, self.B, self.cx, self.cy, self.doffs = Q[2, 3], 1 / Q[3, 2], -Q[0, 3], -Q[1, 3], Q[3, 3] / Q[
                3, 2]

            self.map1_l, self.map2_l = cv2.fisheye.initUndistortRectifyMap(self.intr_l, self.distort_l, R1, P1,
                                                                           new_size, cv2.CV_32F)
            self.map1_r, self.map2_r = cv2.fisheye.initUndistortRectifyMap(self.intr_r, self.distort_r, R2, P2,
                                                                           new_size, cv2.CV_32F)
            # print(f"self.intr_l= {self.intr_l}")
            # print(f"self.intr_r= {self.intr_r}")
            # print(f"self.distort_l= {self.distort_l}")
            # print(f"self.distort_r= {self.distort_r}")
            print(f"size= {size}")
            print(f"new_size= {new_size}")
            print(f"self.Focal={self.Focal}, self.cx={self.cx}, self.cy={self.cy}")
        else:
            raise NotImplementedError
    
    # 矫正
    def rectify_img(self, img_l_path:str, img_r_path:str) -> tuple:
        """
        Rectifies left and right images using precomputed rectification maps.

        Parameters:
            img_l_path (str): File path to the left image.
            img_r_path (str): File path to the right image.

        Returns:
            tuple: A tuple containing the rectified left and right images as numpy arrays.
        """
        if self.map1_l is None:
            self.calc_map(rect_w=self.use_width, rect_h=self.use_height)
        img_l_rectified = img_r_rectified = None
        if os.path.exists(img_l_path) and os.path.exists(img_r_path):
            # img_l = cv2.imread(img_l, cv2.IMREAD_GRAYSCALE)
            # img_r = cv2.imread(img_r, cv2.IMREAD_GRAYSCALE)
            img_l = cv2.imread(img_l_path, cv2.IMREAD_COLOR)
            img_r = cv2.imread(img_r_path, cv2.IMREAD_COLOR)
            # img_l = cv2.imdecode(np.fromfile(img_l, dtype=np.uint8), -1)
            # img_r = cv2.imdecode(np.fromfile(img_r, dtype=np.uint8), -1)
            img_l_rectified = cv2.remap(img_l, self.map1_l, self.map2_l, cv2.INTER_LINEAR)
            img_r_rectified = cv2.remap(img_r, self.map1_r, self.map2_r, cv2.INTER_LINEAR)
            return img_l_rectified, img_r_rectified
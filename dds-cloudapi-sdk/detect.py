# 1. Initialize the client with your API token.
from dds_cloudapi_sdk.tasks.v2_task import create_task_with_local_image_auto_resize
from dds_cloudapi_sdk import Config
from dds_cloudapi_sdk import Client
from dds_cloudapi_sdk.visualization_util import visualize_result
import os
import json
import argparse

class Predict:
    def __init__(self, args):
        self.args = args
        self.config = Config(self.args.token)
        self.client = Client(self.config)
        self.use_category = self.args.use_category
        self.datasets_root_path = self.args.datasets_root_dir
        self.datasets_path = [os.path.join(self.datasets_root_path, dataset_path) for dataset_path in os.listdir(self.datasets_root_path) \
            if os.path.isdir(os.path.join(self.datasets_root_path, dataset_path))]
        print(f"self.datasets_path = {self.datasets_path}")
        self.use_single = self.args.single_detect
        

    def create_foldar(self, foldar_name:str, clear_history:bool=True):
        if os.path.isdir(foldar_name):
            if clear_history:
                os.system(f"rm -rf {foldar_name}")
        os.system(f"mkdir -p {foldar_name}")

    def save_json(self, data:dict, filename:str, save_dir:str):
        name = filename.split(".")[0]
        if not os.path.isdir(save_dir):
            os.system(f"mkdir -p {save_dir}")
        save_path = os.path.join(save_dir , f"{name}.json")
        with open(save_path, 'w') as json_file:
            json.dump(data, json_file, indent=4)

    def inference(self):
        for dataste_path in self.datasets_path:
            save_result_dir = os.path.join(dataste_path, "result")
            print(f"save_result_dir={save_result_dir}")
            visual_dir = os.path.join(save_result_dir, "roi_area", "roi_area_visual")
            json_dir = os.path.join(save_result_dir, "roi_area", "roi_area_json")
            self.create_foldar(visual_dir)
            self.create_foldar(json_dir)
            dataset_image_path = os.path.join(save_result_dir, "images")
            for image_name in os.listdir(dataset_image_path):
                if "left" in  image_name:
                    image_path = os.path.join(dataset_image_path, image_name)
                    task = create_task_with_local_image_auto_resize(
                        api_path="/v2/task/dinox/detection",
                        api_body_without_image={
                            "model": "DINO-X-1.0",
                            # "image": infer_image_url, # not needed for local image
                            "prompt": {
                                "type": "universal",
                                # "text": "wolf.dog.butterfly"
                                # "universal": 1
                            },
                            "targets": ["mask", "bbox"],
                            "bbox_threshold": 0.5,
                            "iou_threshold": 0.8
                        },
                        image_path=image_path
                        )

                    self.client.run_task(task)
                    print(task.result)

                    GREEN = '\033[94m'
                    print(f"{GREEN}=====detect information=====")
                    for info in task.result["objects"]:
                        bbox = info["bbox"]
                        category = info["category"]
                        if category in self.use_category:
                            score = info["score"]
                            print(f"bbox = {bbox}")
                            print(f"category = {category}")
                            print(f"score = {score}")
                            print("================================")

                    # 5. Visualize the result to local image.
                    visualize_result(image_path=image_path, result=task.result, output_dir=visual_dir)
                    self.save_json(task.result, image_name, json_dir)

                    # 每个数据集使用单张照片
                    if self.use_single:
                        break

def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Dinox detect',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--token', type=str, default="b705b99ec89dec76e74b0eb91905adb8", 
                        help='API token')
    parser.add_argument('--datasets_root_dir', type=str, default="/mnt/disk1/chao04.chen/horizon/d-robotics/datasets_test", 
                        help='Directory all camera images with timestamp filenames')
    parser.add_argument('--use_category', nargs='+', type=str, default=["box", "carton"], 
                        help='ROI category')
    parser.add_argument('--single_detect', type=bool, default=True,
                        help='Detect single image ') 
    return parser.parse_args()


def main():
    args = parse_arguments()
    predict = Predict(args)
    predict.inference()

if __name__ == "__main__":
    main()
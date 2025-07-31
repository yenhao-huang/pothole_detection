from kaggle.api.kaggle_api_extended import KaggleApi
import os

api = KaggleApi()
api.authenticate()

save_path = "data"
os.makedirs(save_path, exist_ok=True)

api.dataset_download_files("andrewmvd/pothole-detection", path=save_path, unzip=True)

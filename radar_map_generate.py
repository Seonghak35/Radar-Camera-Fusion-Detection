# --------- Author: songhee-cho --------- #
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

# Global settings
FEATURES_LIST = ['range', 'elevation','doppler', 'power', 'u', 'v']
RESOLUTION = 320

# Original image resolution
ORIGINAL_WIDTH = 1920
ORIGINAL_HEIGHT = 1080

# Scaling calculation based on original resolution
X_SCALE = ORIGINAL_WIDTH / RESOLUTION
Y_SCALE = ORIGINAL_HEIGHT / RESOLUTION

# Paths (customizable)
RADAR_ROOT = "/workspaces/Radar-Camera-Fusion-Detection/WaterScenes/sample_dataset/radar"  # Your own radar csv path
SAVE_RADAR_MAP_ROOT = "/workspaces/Radar-Camera-Fusion-Detection/WaterScenes/sample_dataset/radar/REVP_map" # New dir to save npz

def load_radar_data(radar_root, resolution, x_scale, y_scale):
    radar_files = os.listdir(radar_root)
    print(f"Found {len(radar_files)} radar files.")

    for file in tqdm(radar_files):
        example_file = os.path.join(radar_root, file)
        example_radar_points = pd.read_csv(example_file)[FEATURES_LIST].to_numpy()
        
        # Initialize radar map (range, doppler, rcs)
        example_radar_map = np.zeros((len(FEATURES_LIST) - 2, resolution, resolution))

        for channel in range(len(FEATURES_LIST) - 2):
            for line in example_radar_points:
                try:
                    row_index = int(line[-2] / x_scale)
                    column_index = int(line[-1] / y_scale)

                    # Handle conflicts by moving to the previous row
                    if example_radar_map[channel][row_index][column_index] != 0 and row_index >= 1:
                        row_index -= 1
                    
                    example_radar_map[channel][row_index][column_index] = line[channel]
                except:
                    continue

        # Adjust the order of axes for saving
        example_radar_map = example_radar_map.transpose(0, 2, 1)
        yield file, example_radar_map

def save_radar_data(save_root, file_name, radar_map):
    # Create direcotry if it doesn't exist
    save_path = os.path.join(save_root, file_name)
    dir_path = os.path.dirname(os.path.join(save_root, file_name))
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    np.savez_compressed(save_path, radar_map)

def main():
    if not os.path.exists(SAVE_RADAR_MAP_ROOT):
        os.makedirs(SAVE_RADAR_MAP_ROOT)

    for file, radar_map in load_radar_data(RADAR_ROOT, RESOLUTION, X_SCALE, Y_SCALE):
        save_file = file.replace('.csv', '.npz')
        save_radar_data(SAVE_RADAR_MAP_ROOT, save_file, radar_map)

if __name__ == "__main__":
    main()
import numpy as np 
import torch 
import yaml 
import tal 
import os
import argparse

from tqdm import tqdm
from torch import optim as optimizers

from model.scene import Scene
from model.nerf import NLOSNeRF
from model.loss import NeTFLoss


if __name__ == "__main__":
    
    n_epochs_default = 50
    device_uri_gpu = "cuda:1"
    device_uri_cpu = "cpu"
    batch_time_size_default = 4
    n_hemisphere_bins = 32
    delta_az = torch.pi / n_hemisphere_bins
    delta_col = torch.pi / n_hemisphere_bins
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_hdf5", action="store", type=str, required=True)
    parser.add_argument("--path_yaml", action="store", type=str, required=True)
    parser.add_argument("--epochs", action="store", type=int, default=n_epochs_default,
                        help="Number of epochs for trainning")
    parser.add_argument("--batch_time_bins", action="store", type=int, default=batch_time_size_default,
                        help="batch of time bins for ground truth")
    
    args = parser.parse_args()
    epochs = args.epochs
    batch_time_size = args.batch_time_bins
    path = args.path_hdf5
    path_yaml = args.path_yaml
    
    gt_data = tal.io.read_capture(path)
    
    
    print(f"Initialization of model and allocation on GPU: {device_uri_gpu}")
    
    scene = Scene(sensor_x=sensor_width, sensor_y=sensor_height, scale=scale_rw, device=device_uri_cpu)
    H_time_dim = gt_data.H_format.time_dim()
    t_max = gt_data.H.shape[H_time_dim]
    time_range = torch.arange(gt_data.H.shape[H_time_dim])
    radius_bins = torch.linspace(start=0, end=t_max, steps=t_max) * delta_bins / 2
    model = NLOSNeRF(device=device_uri_gpu).to(device=device_uri_gpu)
    gt_H = torch.from_numpy(gt_data.H)
    
    
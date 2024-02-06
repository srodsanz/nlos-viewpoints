import argparse
import os
import tal
import logging
import torch

from tqdm import tqdm
from torch.optim import optimizer

from model import *
    
from model.nerf import NLOSNeRF
from model.render import Renderer
from model.volume import Volume


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--path", action="store", type=str, required=True)
    parser.add_argument("--epochs", action="store", type=int, default=EPOCHS_DEFAULT,
                        help="Number of epochs for trainning")
    parser.add_argument("--delta", action="store", type=float, default=DELTA_METERS_DEFAULT,
                        help="Delta bin discriminant for meters discretization")
    parser.add_argument("--device", action="store", type=int, default=DEVICE_DEFAULT,
                        help="Device URI for instantiation")
    parser.add_argument("--tbins", action="store", type=int, default=TIME_BINS_DEFAULT,
                        help="Number of time bins for input transients")
    parser.add_argument("--loss", action="store", type=str, choices=LOSS_LITERALS,
                        help="String identifier for loss function")
    parser.add_argument("--scale", action="store", type=int, default=SCALE_RELAY_WALL_DEFAULT,
                        help="Scale factor of relay wall dimension")
    parser.add_argument("--sensor_width", action="store", type=int, default=SENSOR_WIDTH_DEFAULT,
                        help="Sensor width for relay wall generation")
    parser.add_argument("--sensor_height", action="store", type=int, default=SENSOR_HEIGHT_DEFAULT,
                        help="Sensor height for relay wall definition")
    parser.add_argument("--speed_wave", action="store", type=float, default=SPEED_OF_LIGHT,
                        help="Speed of referenced wave on Imaging problem")
    parser.add_argument("--volume_bins", action="store", type=int, default=VOLUME_SAMPLES_DEFAULT,
                        help="Number of samples for volume bins")
    parser.add_argument("--laser_width", action="store", type=int, default=LASER_WIDTH_DEFAULT,
                        help="Laser width dimension of input grid")
    parser.add_argument("--laser_height", action="store", type=int, default=LASER_HEIGHT_DEFAULT,
                        help="Laser height dimension of the input grid")

    args = parser.parse_args()
    path = args.path
    n_epochs = args.epochs
    delta_bins = args.delta 
    device = args.device
    speed_of_light = args.speed_wave 
    n_t_bins = args.tbins 
    loss = args.loss
    scale = args.scale
    sensor_width = args.sensor_width
    sensor_heigh = args.sensor_height
    n_vol_bins = args.volume_bins
    step_update = 5
    
    data = tal.io.read_capture(path)
    ground_truth_H = torch.from_numpy(data.H)
    model = NLOSNeRF().to(device=device)
    relay_wall = Renderer.generate_relay_wall(sensor_x=sensor_width, sensor_y=sensor_heigh, scale=scale)
    
    with tqdm(total=n_epochs, position=0) as pbar:
        
        for i in range(n_epochs):
            
            pbar.update()
            
            if not i % step_update:
                pbar.set_description(f"Loss value: {loss} at epoch batch : {i}")
            
            
                    
        
    
    
    
    
    
    
    
    
    


    




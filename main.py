import torch 
import os 
import tal 
import argparse
import time

from tqdm import tqdm
from torch import optim as optimizers
from matplotlib import pyplot as plt

from model.scene import Scene
from model.nerf import NLOSNeRF
from model.loss import NeTFLoss
from model.render import Renderer


if __name__ == "__main__":
    
    n_gradient_steps = 30
    n_epochs_default = n_gradient_steps*1024
    n_hemisphere_bins = 32
    scale_rw = 1
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", action="store", type=str, required=True,
                        help="Path of y-tal hdf5 path of ground truth data")
    parser.add_argument("--path_model", action="store", type=str, required=True,
                        help="Model weights")
    parser.add_argument("--device_uri_gpu", action="store", type=str,
                        help="URI for GPU device")
    parser.add_argument("--path_intermediate_results", action="store", type=str, required=True,
                        help="Intermediate results path")
    
    args = parser.parse_args()
    path = args.path 
    path_model = args.path_model
    device_uri_gpu = args.device_uri_gpu
    path_intermediate_results = args.path_intermediate_results
    
    assert os.path.exists(path) and os.path.isfile(path), f"Path {path} does not exist"
    
    gt_data = tal.io.read_capture(path)
    delta_m_meters = gt_data.delta_t
    gt_H = torch.from_numpy(gt_data.H)
    gt_H = torch.moveaxis(gt_H, source=0, destination=-1)
    sensor_width, sensor_height, t_max = gt_H.shape
    device = torch.device(device_uri_gpu if torch.cuda.is_available() else "cpu")
    
    print(f"Initialization of model and sampling of LF")
    
    scene = Scene(sensor_x=sensor_width, sensor_y=sensor_height, scale=scale_rw)
    model = NLOSNeRF().to(device=device)
    adam = optimizers.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    
    print("Init sampling of LF")
    
    t0 = time.time()
    
    global_lf = scene.generate_light_field(time_start=0, time_end=t_max,
                                        n_spherical_coarse_bins=n_hemisphere_bins,
                                        delta_m_meters=delta_m_meters)[..., [0, 1, 2, 4, 5]]
    
    t1 = time.time()
    
    print(f"Sampling of LF achieved in : {t0-t1} seconds")
    
    with tqdm(range(n_epochs_default)) as pbar:
        
        adam.zero_grad()
        global_loss = 0
        
        for i in range(n_epochs_default):
            
            t_idx = i % (t_max - 1) + 1
            pred_H = torch.zeros((sensor_width, sensor_height, 1))
            sampled_gt_H = gt_H[:, :, t_idx:t_idx+1]
            batch_lf = global_lf[:, :, t_idx:t_idx+1, :, :, :]
            col_bins = batch_lf[..., -1]
            
            batch_lf_pe = model.fourier_encoding(batch_lf).to(device=device)
            pred_volume_albedo = model(batch_lf_pe).cpu()
            
            pred_transient = Renderer.render_quadrature_transient(
                predicted_volume_albedo=pred_volume_albedo,
                delta_m_meters=delta_m_meters,
                time_start=t_idx,
                time_end=t_idx+1,
                col_bins=col_bins,
                n_spherical_coarse_bins=n_hemisphere_bins
            )
            
            loss = NeTFLoss.squared_error(pred_transient, sampled_gt_H)
            global_loss += loss
            loss.backward()
            
            if i % t_max == 0 and i > 0:
                adam.step()
                adam.zero_grad()
                global_loss = 0
                
            if i % t_max == 0:
                
                with torch.no_grad():
                    fig, ax = plt.subplots()
                    xv = torch.linspace(start=-1, end=1, steps=32)
                    yv = torch.linspace(start=-1, end=1, steps=32)
                    z = 0.5*torch.ones((32, 32))
                    X, Y = torch.meshgrid(xv, yv)
                    stack_pts = torch.stack((X, Y, z), axis=-1)
                    view_dirs = torch.zeros((32, 32, 2))
                    stacked_pts_dirs = model.fourier_encoding(torch.cat((stack_pts, view_dirs), dim=-1))
                    pred = model(stacked_pts_dirs.to(device=device)).cpu()
                    pred_vol = torch.prod(pred, dim=-1)
                    ax.imshow(pred_vol.cpu().detach().numpy())
                    fig.savefig(f"{path_intermediate_results}/result_{i}.png")
                    fig.clear()
                    ax.clear()
                
            
            pbar.update(1)
            pbar.set_description(f"Loss function: {global_loss} at epoch: {i}")
    
    print("Init model saving")
    torch.save(model.state_dict(), path_model)
            
            
            
            
            
            
            
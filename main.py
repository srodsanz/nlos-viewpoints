import torch 
import os 
import tal 
import numpy as np 
import argparse
import time

from tqdm import tqdm
from torch import optim as optimizers
from matplotlib import pyplot as plt

from model.scene import Scene
from model.nerf import NLOSNeRF
from model.loss import NeTFLoss
from model.render import Renderer
from model.context import NeRFContext


if __name__ == "__main__":
        
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", action="store", type=str, required=True,
                        help="Path of y-tal hdf5 path of ground truth data")
    parser.add_argument("--checkpoint", action="store_true",
                        help="Flag to indicate saving checkpoint")
    parser.add_argument("--path_model", action="store", type=str, required=True,
                        help="Model weights")
    parser.add_argument("--device_uri_gpu", action="store", type=str,
                        help="URI for GPU device")
    parser.add_argument("--path_intermediate_results", action="store", type=str, required=True,
                        help="Intermediate results path")
    parser.add_argument("--reload_checkpoint", action="store_true",
                        help="indicate if training over checkpoint model")
    parser.add_argument("--path_checkpoint", action="store",
                        help="path of serialized model for reload")
    parser.add_argument("--epochs", action="store", type=int, required=True,
                        help="Number of iterations over whole transient simulation")
    parser.add_argument("--number_hemisphere_sampling", type=int, required=True,
                        help="Number of sampled hemispheres")
    parser.add_argument("--arg_start", type=float, action="store", default=0,
                        help="Start range or argument sampling")
    parser.add_argument("--arg_end", type=float, action="store", default=torch.pi,
                        help="Argument end bound for angle sampling")
    
    args = parser.parse_args()
    path = args.path 
    path_model = args.path_model
    device_uri_gpu = args.device_uri_gpu
    path_intermediate_results = args.path_intermediate_results
    n_iter = args.epochs
    n_hemisphere_bins = args.number_hemisphere_sampling
    arg_start = args.arg_start
    arg_end = args.arg_end
    
    device = torch.device(device_uri_gpu if torch.cuda.is_available() else "cpu")
    
    assert os.path.exists(path) and os.path.isfile(path), f"Path {path} does not exist"
    
    gt_data = tal.io.read_capture(path)
    
    NeRFContext.from_ytal(gt_data, n_iter=n_iter, n_sampled_hemispheres=n_hemisphere_bins)
    
    print(f"Initialization of model and sampling of LF")
    
    sensor_width, sensor_height, t_max = NeRFContext.sensor_width, NeRFContext.sensor_height, NeRFContext.t_max
    delta_m_meters = NeRFContext.delta_m_meters
    n_epochs_default = NeRFContext.n_iter
    gt_H = NeRFContext.H
    gradient_batch_update = t_max
    
    scene = Scene(sensor_x=sensor_width, sensor_y=sensor_height)
    model = NLOSNeRF().to(device=device)
    adam = optimizers.Adam(model.parameters(), lr=5e-3)
    
    print(f"""
        Init training scheme with the following parameters
        time max = {t_max}
        sensor_width = {sensor_width}
        sensor_height = {sensor_height}
        iterations = {n_epochs_default}
        delta meters = {delta_m_meters}
        number of hemisphere bins = {n_hemisphere_bins}
        gradient updates = {gradient_batch_update}
    """)
    
    
    if args.reload_checkpoint:
        path_checkpoint = args.path_checkpoint
        assert path_checkpoint is not None and os.path.exists(path_checkpoint) and os.path.isfile(path_checkpoint), f"Not provided path"
        state_dict = torch.load(path_checkpoint)
        model.load_state_dict(state_dict["model_state_dict"])
        adam.load_state_dict(state_dict["optimizer_state_dict"])
        model.train()
    
    print("Init sampling of LF")
    
    t0 = time.time()
    
    global_lf = scene.generate_light_field(time_start=0, time_end=t_max, arg_start=arg_start, arg_end=arg_end,
                                        n_spherical_coarse_bins=n_hemisphere_bins,
                                        delta_m_meters=delta_m_meters)[..., [0, 1, 2, 4, 5]]
    
    t1 = time.time()
    
    print(f"Sampling of LF achieved in : {t1 - t0} seconds")
    
    with tqdm(range(n_epochs_default)) as pbar:
        
        adam.zero_grad()
        
        for i in range(n_epochs_default):
            
            if (i+1) % t_max == 0:
                
                with torch.no_grad():
                    global_loss = 0
                    fig, ax = plt.subplots()
                    center = np.array([0, 0, 0.5])
                    dx = 1 / np.sqrt(2)
                    dy = 1 / np.sqrt(2)
                    dz = 1 / np.sqrt(2)
                    xv = torch.linspace(start=center[0]-dx/2, end=center[0]+dx/2, steps=32)
                    yv = torch.linspace(start=center[1]-dy/2, end=center[1]+dy/2, steps=32)
                    zv = torch.linspace(start=center[2]-dz/2, end=center[2]+dz/2, steps=32)
                    X, Y, Z = torch.meshgrid(xv, yv, zv, indexing="ij")
                    stack_pts = torch.stack((X, Y, Z), axis=-1)
                    view_dirs = (torch.pi/2)*torch.ones((32, 32, 32, 2))
                    stacked_pts_dirs = model.fourier_encoding(torch.cat((stack_pts, view_dirs), dim=-1))
                    pred = model(stacked_pts_dirs.to(device=device)).cpu()
                    pred_vol = torch.prod(pred, axis=-1).cpu().detach().numpy()
                    ax.imshow(np.max(pred_vol, axis=-1))
                    fig.savefig(f"{path_intermediate_results}/result_{i}.png")
                    plt.close()
            
            
            t_idx = i % t_max 
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
                arg_start=arg_start,
                arg_end=arg_end,
                col_bins=col_bins,
                n_spherical_coarse_bins=n_hemisphere_bins
            )
            
            loss = NeTFLoss.squared_error(pred_transient, sampled_gt_H) / gradient_batch_update
            loss.backward()
            
            if (i+1) % gradient_batch_update == 0:
                adam.step()
                adam.zero_grad()
            
            pbar.update(1)
            pbar.set_description(f"Loss function: {loss} at epoch: {i}")
            
    
    if args.checkpoint:
        print("Init model saving")
        torch.save(
            {
                "epoch": i,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": adam.state_dict()
            }, path_model
        )            
            
            
            
            
            
            
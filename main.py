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
    parser.add_argument("--path_model", action="store", type=str, required=True,
                        help="Model weights")
    parser.add_argument("--device_gpu", action="store", type=str,
                        help="URI for GPU device")
    parser.add_argument("--path_results", action="store", type=str, required=True,
                        help="Intermediate results path")
    
    #Checkpointing
    parser.add_argument("--reload_checkpoint", action="store_true",
                        help="indicate if training over checkpoint model")
    parser.add_argument("--path_checkpoint", action="store",
                        help="path of serialized model for reload")
    parser.add_argument("--loss", action="store", default=[e.name for e in NeTFLoss],
                        help="loss function: MSE or SE")
    parser.add_argument("--ignore_checkpoint", action="store_true",
                        help="Flag to indicate that model checkpoint is saved")
    
    #Optimization tuning
    parser.add_argument("--epochs", action="store", type=int, required=True,
                        help="Number of iterations over whole transient simulation")
    parser.add_argument("--number_hemisphere_sampling", type=int, required=True,
                        help="Number of sampled hemispheres")
    parser.add_argument("--arg_start", type=float, action="store", default=0,
                        help="Argument lower bound for range of argument sampling")
    parser.add_argument("--arg_end", type=float, action="store", default=torch.pi,
                        help="Argument upper bound for angle sampling")
    parser.add_argument("--number_gradient_updates", type=int, action="store",
                        help="Number of gradient accumulation steps")
    parser.add_argument("--lr", action="store", type=float, default=5e-3,
                        help="Learning rate")
    
    #Sampling
    parser.add_argument("--light_cone_sampling", action="store_true", 
                        help="Indicate if spatial sampling is applied over the projected area")
    parser.add_argument("--importance_fine_sampling", action="store_true",
                        help="Indicate if using importance sampling over")
    parser.add_argument("--illumination_offset", action="store", type=int, default=1,
                        help="Offset on illumination points sampling")
    parser.add_argument("--importance_sampling", action="store_true", 
                        help="Flag to indicate if performing importance sampling")
    
    #Model options
    parser.add_argument("--length_pe", action="store", type=int, default=5,
                        help="Length of positional encoding of frequencies")
    parser.add_argument("--no_pe", action="store_false",
                        help="No positional encoding")
    
    #Reproducibility
    parser.add_argument("--seed", action="store", type=int, default=0,
                        help="Seed of initialization")
    
    args = parser.parse_args()
    path = args.path 
    path_model = args.path_model
    device_uri_gpu = args.device_gpu
    path_intermediate_results = args.path_results
    n_iter = args.epochs
    n_hemisphere_bins = args.number_hemisphere_sampling
    arg_start = args.arg_start
    arg_end = args.arg_end
    seed = args.seed
    loss_id = args.loss
    lr = args.lr
    number_gradient_updates = args.number_gradient_updates
    lcs_sampling_step = 2
    length_embeddings = args.length_pe
    ill_offset = args.illumination_offset
        
    device = torch.device(device_uri_gpu if torch.cuda.is_available() else "cpu")
    loss_func = NeTFLoss.func_from_id(loss_id)
    
    assert os.path.exists(path) and os.path.isfile(path), f"Path {path} does not exist"
    
    gt_data = tal.io.read_capture(path)
    
    print("Compensation of radiation for radiance computation decay")
    
    tal.reconstruct.compensate_laser_cos_dsqr(gt_data)
    
    NeRFContext.from_ytal(gt_data, n_iter=n_iter, n_sampled_hemispheres=n_hemisphere_bins)
    
    print(f"Initialization of model and sampling of LF")
    
    sensor_width, sensor_height, t_max = NeRFContext.sensor_width, NeRFContext.sensor_height, NeRFContext.t_max
    delta_m_meters = NeRFContext.delta_m_meters
    n_epochs_default = NeRFContext.n_iter
    gt_H = NeRFContext.H
    
    scene = Scene(sensor_x=sensor_width, sensor_y=sensor_height)
    model = NLOSNeRF(positional_encoding=args.no_pe, length_embeddings=length_embeddings).to(device=device)
    adam = torch.optim.Adam(model.parameters(), lr=lr)
    
    sensor_scale = scene.get_sensor_scale()
    
    print(f"Manual seed set to: {seed}")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    
    print(f"""
        Init training scheme with the following parameters
        time max = {t_max}
        sensor_width = {sensor_width}
        sensor_height = {sensor_height}
        iterations = {n_epochs_default}
        delta meters = {delta_m_meters}
        number of hemisphere bins = {n_hemisphere_bins}
        arg start = {arg_start}
        arg end = {arg_end}
        path results = {path_intermediate_results}
        path input = {path}
        path to save model = {path_model}
        loss function = {loss_func.__name__}
        Shape ground truth H = {gt_H.shape}
        Learning rate = {lr}
        Reloading = {args.reload_checkpoint}
        Ignore checkpoint = {args.ignore_checkpoint}
        Light cone sampling = {args.light_cone_sampling}
        Light come sampling step = {lcs_sampling_step}
        gradient updates = {number_gradient_updates}
    """)
    
    
    if args.reload_checkpoint:
        path_checkpoint = args.path_checkpoint
        assert path_checkpoint is not None and os.path.exists(path_checkpoint) and os.path.isfile(path_checkpoint), f"Not provided path"
        state_dict = torch.load(path_checkpoint)
        model.load_state_dict(state_dict["model_state_dict"])
        adam.load_state_dict(state_dict["optimizer_state_dict"])
    
    r_max = delta_m_meters * t_max / 2
    rw = scene.relay_wall()
    
    x_min, x_max = torch.min(rw[..., 0]) - r_max, torch.max(rw[..., 0]) + r_max
    y_min, y_max = torch.min(rw[..., 1]) - r_max, torch.max(rw[..., 1]) + r_max
    z_min, z_max = 0, r_max
    sensor_resolution = sensor_width * sensor_height
    
    with tqdm(range(n_epochs_default)) as pbar:
        
        adam.zero_grad()
        cum_time_epoch = 0
        
        for i in range(n_epochs_default):
            
            t0 = time.time()
            
            if args.light_cone_sampling and i % lcs_sampling_step == 0:
                width_idx, height_idx = NeRFContext.sample_mkw_light_cone()
            
            else:
                sensor_idx = i % sensor_resolution
                width_idx = sensor_idx // sensor_width
                height_idx = sensor_idx % sensor_height
            
            batch_lf = scene.generate_light_field(sensor_width_idx=width_idx,
                        sensor_height_idx=height_idx,
                        w_offset=ill_offset,
                        h_offset=ill_offset,
                        time_start=0,
                        time_end=t_max,
                        arg_start=arg_start,
                        arg_end=arg_end,
                        n_spherical_coarse_bins=n_hemisphere_bins,
                        delta_m_meters=delta_m_meters
            )[..., [0, 1, 2, 4, 5]]
            
            sampled_gt_H = gt_H[width_idx:width_idx+ill_offset, height_idx:height_idx+ill_offset, ...]
            
            col_bins = batch_lf[..., -1] # TODO: independent of light field format
            
            batch_lf_pe = model.fourier_encoding(batch_lf, 
                                                x_min=x_min,
                                                x_max=x_max,
                                                y_min=y_min, 
                                                y_max=y_max,
                                                z_min=z_min,
                                                z_max=z_max).to(device=device)
            
            pred_volume_albedo = model(batch_lf_pe).cpu()
            
            pred_transient = Renderer.render_quadrature_transient(
                predicted_volume_albedo=pred_volume_albedo,
                delta_m_meters=delta_m_meters,
                time_start=0,
                time_end=t_max,
                arg_start=arg_start,
                arg_end=arg_end,
                col_bins=col_bins,
                n_spherical_coarse_bins=n_hemisphere_bins
            )
            
            loss = loss_func(transient_pred=pred_transient, transient_gt=sampled_gt_H) / number_gradient_updates
            loss.backward()
            
            if (i+1) % number_gradient_updates == 0:
                adam.step()
                adam.zero_grad()
            
            if (i+1) % (t_max) == 0:
                
                with torch.no_grad():
                    fig, ax = plt.subplots()
                    center = np.array([0, 0, 0.5])
                    dx = 1 / np.sqrt(2)
                    dy = 1 / np.sqrt(2)
                    dz = 1 / np.sqrt(2)
                    xv = torch.linspace(start=center[0]-(dx/2), end=center[0]+(dx/2), steps=sensor_width)
                    yv = torch.linspace(start=center[1]-(dy/2), end=center[1]+(dy/2), steps=sensor_width)
                    zv = torch.linspace(start=center[2]-(dz/2), end=center[2]+(dz/2), steps=sensor_width)
                    X, Y, Z = torch.meshgrid(xv, yv, zv,  indexing="ij")
                    stack_pts = torch.stack((Y, X, Z), axis=-1)
                    view_dirs = (torch.pi/2)*torch.ones((sensor_width, sensor_width, sensor_width, 2))
                    stacked_pts_dirs = model.fourier_encoding(torch.cat((stack_pts, view_dirs), dim=-1),
                                                            x_min=x_min,
                                                            x_max=x_max,
                                                            y_min=y_min, 
                                                            y_max=y_max,
                                                            z_min=z_min,
                                                            z_max=z_max)
                    pred = model(stacked_pts_dirs.to(device=device)).cpu()
                    pred_vol = torch.prod(pred, axis=-1).cpu().detach().numpy()
                    ax.imshow(np.max(pred_vol, axis=-1), cmap="hot")
                    fig.savefig(f"{path_intermediate_results}/result_{i+1}.png")
                    plt.close()
            
            cum_time_epoch += time.time() - t0
            
            pbar.set_description(f"Loss function: {loss} at epoch: {i}")
            pbar.update(1)
            
    
    if not args.ignore_checkpoint:
        print("Init model saving")
        torch.save(
            {
                "number_epochs": n_epochs_default,
                "sensor_width": sensor_width,
                "sensor_height": sensor_height,
                "n_hemisphere_coarse_bins": n_hemisphere_bins,
                "time_bins": t_max,
                "illumination_offset": ill_offset,
                "avg_time_epoch": cum_time_epoch / n_epochs_default,
                "learning_rate": lr,
                "batch_gradients_size": number_gradient_updates,
                "torch_seed": torch.seed(), 
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": adam.state_dict()
            }, path_model
        )
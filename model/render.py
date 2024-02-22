import numpy as np 
import torch

from .format import LightFFormat
from .scene import Scene

class Renderer:
    """
    Rendering class on volume rendering techniques
    """
    
    def __init__(self, focal, device):
        """
        Constructor
        """
        assert (device.startswith("cuda") and torch.cuda.is_available()) or (device == "cpu"), \
            f"The specified device: {device} is not available or unknown"
        
        #TODO: Techniques for hierarchical sampling on PDF for geometry, maybe with weights over bounded volume
        self.focal = focal
        self.device = device
        
    def render_transient(self, nerf_fn, 
                        lf_sampled_center,
                        lf_format: LightFFormat = LightFFormat.LF_X0_Y0_R_A_C_6
        ):
        
        """
        Render transient with quadrature rule
        Sampled center-wise

        Args:
            lf_batch (torch.Tensor): _description_
            lf_format (LightFFormat, optional): _description_. Defaults to LightFFormat.LF_X0_Y0_R_A_C_6.
        """
        assert lf_sampled_center.shape[-1] == 6 and lf_sampled_center.dim() == 4, f"Passed sampling on light field has incorrect shapes"
        
        n_radius_bins = lf_sampled_center.shape[0]
        pred_H = torch.zeros((n_radius_bins)).to(device=self.device)
        radius_bins = lf_sampled_center[:, 0, 0, 0]
        
        if lf_format == LightFFormat.LF_X0_Y0_R_A_C_6:
            az, col = lf_sampled_center[0, :, 0, -2], lf_sampled_center[0, 0, :, -1]
        else:
            col, az = lf_sampled_center[0, :, 0, -2], lf_sampled_center[0, 0, :, -1]
        
        delta_az = (az[-1] - az[0]) / az.shape[0]
        delta_col = (col[-1] - col[0]) / col.shape[0]
        
        #TODO: To prevent OOM pass into torch model each radius ... maybe possible to vectorize and avoid loop in some other way ?
        pred = nerf_fn(lf_sampled_center)
        sampled_pts = lf_sampled_center.reshape(-1, 6)
        colatitude = sampled_pts[:, -2] if lf_format == LightFFormat.LF_X0_Y0_R_C_A_6 else sampled_pts[:, -1]
        pred_H = (delta_az * delta_col / radius_bins ** 2) * torch.sum(torch.prod(pred, dim=-1)*torch.sin(colatitude), dim=(-1, -2))        
        
        return pred_H
    
            
    def get_rays(self, H, W,
                camera2world: torch.Tensor
        ):
        """
        Sample rays from pinhole camera model

        Args:
            H (_type_): _description_
            W (_type_): _description_
            camera2world (torch.Tensor): _description_
        """
        assert H > 0 and W > 0, f"Width {W} and height {H} should be > 0"
        h = torch.arange(H).to(device=self.device)
        w = torch.arange(W).to(device=self.device)
        focal = self.focal
        ww, hh = torch.meshgrid(h, w, indexing="xy")
        dirs = torch.stack(((ww - 0.5*W) / focal, -(hh - 0.5*H) / focal, -torch.ones_like(hh)), axis=-1)
        rays_o = torch.broadcast_to(camera2world[:3, -1], dirs.shape)
        rays_d = torch.sum(dirs[..., None, :] * camera2world[:3, :3], axis=-1)
        
        return rays_o, rays_d
    
    def ndc_rays(self, H, W, focal, near,
                    rays_o, rays_d):
        """
        Convert to NDC coordinate system

        Args:
            H (_type_): _description_
            W (_type_): _description_
            focal (_type_): _description_
            near (_type_): _description_
            rays_o (_type_): _description_
            rays_d (_type_): _description_
        """
        
        # Shift ray origins to near plane
        t = -(near + rays_o[..., 2]) / rays_d[..., 2]
        rays_o = rays_o + t[..., None] * rays_d

        # Projection
        o0 = -1./(W/(2.*focal)) * rays_o[..., 0] / rays_o[..., 2]
        o1 = -1./(H/(2.*focal)) * rays_o[..., 1] / rays_o[..., 2]
        o2 = 1. + 2. * near / rays_o[..., 2]

        d0 = -1./(W/(2.*focal)) * \
            (rays_d[..., 0]/rays_d[..., 2] - rays_o[..., 0]/rays_o[..., 2])
        d1 = -1./(H/(2.*focal)) * \
            (rays_d[..., 1]/rays_d[..., 2] - rays_o[..., 1]/rays_o[..., 2])
        d2 = -2. * near / rays_o[..., 2]

        rays_o = torch.stack([o0, o1, o2], -1)
        rays_d = torch.stack([d0, d1, d2], -1)

        return rays_o, rays_d
        
    def render_rays(self, 
                    nerf_fn, H, W, rays_o, rays_d, 
                    near, far, 
                    samples_per_ray,
                    batch_size=32
        ):
        """_summary_

        Args:
            rays_o (_type_): _description_
            rays_d (_type_): _description_
            near (_type_): _description_
            far (_type_): _description_
            samples_per_ray (_type_): _description_
        """
        z = torch.linspace(start=near, end=far, steps=samples_per_ray).expand((H, W, samples_per_ray)) \
            .to(device=self.device)
        
        pts = rays_o.unsqueeze(-2) + rays_d.unsqueeze(-2) * z.unsqueeze(-1)
        viewing_dirs = rays_d / torch.linalg.norm(rays_d, dim=-1)
        pts = pts.reshape(-1, 3)
        viewing_dirs = viewing_dirs.reshape(-1, 3)
        
        assert pts.shape[-1] == viewing_dirs.shape[-1] == 3 and pts.shape[0] == viewing_dirs.shape[0], \
            f"Incorrect shapes in implementation of rays and origins"
        
        field_xyz = torch.cat((pts, viewing_dirs), dim=1)
        n_pts = pts.shape[0]
        raw = torch.zeros((n_pts, 6))
        batch_indexes = [i for i in range(n_pts) if i % batch_size == 0]
        
        for i, batch_idx in enumerate(batch_indexes):
            
            batch = field_xyz[batch_indexes[i-1]: batch_idx] if i >= 1 else field_xyz[:batch_idx]
            pred = nerf_fn(batch)
            if i == 0:
                raw = pred
            else:
                raw = torch.cat((raw, pred), dim=0)
        
        raw = raw.view((H, W, samples_per_ray, 2))
        volume_density, albedo = raw[..., 0], raw[..., 1]
        dists = torch.diff(z, dim=-1)
        transmit = torch.exp(
            -torch.cumsum(volume_density.unsqueeze(-1) * dists.unsqueeze(-2), dim=-1)
        )
        volume = volume_density * transmit
        albedo = albedo * transmit
        intensity = volume_density * albedo * transmit
        
        intensity_map = torch.sum(intensity.unsqueeze(-1) * dists.unsqueeze(-2), dim=-1)
        albedo_map = torch.sum(albedo.unsqueeze(-1) * dists.unsqueeze(-2), dim=-1)
        volume_map = torch.sum(volume.unsqueeze(-1) * dists.unsqueeze(-2), dim=-1)
        
        return intensity_map, albedo_map, volume_map
    
    def render(self, nerf_fn, H, W, camera2world, 
            near, far, samples_per_ray):
        """_summary_

        Args:
            nerf_fn (_type_): _description_
            H (_type_): _description_
            W (_type_): _description_
            camera2world (_type_): _description_
            near (_type_): _description_
            far (_type_): _description_
            samples_per_ray (_type_): _description_
        """
        rays_o, rays_d = self.get_rays(H=H, W=W, camera2world=camera2world)
        rays_o, rays_d = self.ndc_rays(H=H, W=W, focal=self.focal, near=near, rays_o=rays_o, rays_d=rays_d)
        return self.render_rays(near_fn=nerf_fn,
                    H=H, W=W, rays_o=rays_o, rays_d=rays_d, near=near, far=far,
                    samples_per_ray=samples_per_ray)        
import numpy as np 
import torch

from .scene import Scene

class Renderer:
    """
    Rendering class on volume rendering techniques
    """
    
    def __init__(self, focal):
        """
        Constructor
        """        
        self.focal = focal
    
    @classmethod
    def render_quadrature_transient(cls,
                    predicted_volume_albedo,
                    delta_m_meters,
                    time_start,
                    time_end,
                    arg_start,
                    arg_end,
                    col_bins,
                    n_spherical_coarse_bins
    ):
        """_summary_

        Args:
            predicted_volume_albedo (_type_): _description_
            col_bins (_type_): _description_
            radius_bins (_type_): _description_
            n_spherical_coarse_bins (_type_): _description_
            lf_form (_type_): _description_
        """
        assert col_bins.dim() == 5, f"Provided colatitude bins does not have same shape as LF tensor"
        assert arg_start <= arg_end, f"Input arguments start = {arg_start} and end = {arg_end}"
        
        radius_bins = torch.arange(start=time_start, end=time_end) * delta_m_meters / 2
        
        if time_start == 0:
            radius_bins[0] = radius_bins[0] + delta_m_meters
        
        delta_az = (arg_end - arg_start) / n_spherical_coarse_bins
        delta_col = (arg_end - arg_start) / n_spherical_coarse_bins
        scaling = (delta_az * delta_col)
        density = torch.sum(torch.prod(predicted_volume_albedo, axis=-1) * torch.sin(col_bins), dim=(-2, -1)) / radius_bins ** 2
        
        return scaling * density
    
    @classmethod
    def render_mc_transient(cls,
                    predicted_volume_albedo,
                    pdf,
                    delta_m_meters, 
                    time_start,
                    time_end
        ):
        """_summary_

        Args:
            predicted_volume_albedo (_type_): _description_
            pdf (_type_): _description_
            radius_bins (_type_): _description_
        """
        radius_bins = torch.arange(start=time_start, end=time_end) * delta_m_meters / 2
        if time_start == 0:
            radius_bins[0] = radius_bins[0] + 1e-5
        
        density = torch.sum(torch.prod(predicted_volume_albedo, axis=-1) / pdf, dim=(-1, -2))
        return density / radius_bins ** 4
    
    def get_rays(self, 
                H, 
                W,
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
    
    def ndc_rays(self, 
                H, 
                W, 
                focal, 
                near,
                rays_o, rays_d
    ):
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
                    nerf_fn, 
                    H, W, rays_o, rays_d, 
                    near, far, 
                    samples_per_ray,
                    batch_size=32
        ):
        """
        Rander rays routine 

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
        
        volume = volume_density 
        albedo = albedo 
        intensity = volume_density * albedo 
        
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
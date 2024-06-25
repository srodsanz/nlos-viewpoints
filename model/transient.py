import numpy as np 
import torch

from .scene import Scene

class Transient:
    """
    Rendering class on volume rendering techniques
    """
    
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
            radius_bins[0] = radius_bins[0] + 1e-4
        
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
            radius_bins[0] = radius_bins[0] + 1e-4
        
        density = torch.sum(torch.prod(predicted_volume_albedo, axis=-1) / pdf, dim=(-1, -2))
        return density / radius_bins ** 4
    
    
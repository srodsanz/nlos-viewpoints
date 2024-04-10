
import torch

from tal.io.capture_data import NLOSCaptureData
from tal.enums import HFormat

from .format import BBox


class NeRFContext:
    
    n_sampled_hemispheres = None
    t_max = None 
    n_iter = None
    sensor_width = None
    sensor_height = None
    H = None
    delta_m_meters = None
    
    @classmethod
    def from_ytal(cls, 
                data: NLOSCaptureData,
                n_iter: int,
                n_sampled_hemispheres
        ):
        """
        Import NeRF Context from simulated transient

        Args:
            data (NLOSCaptureData): _description_
        """
        cls.H = torch.from_numpy(data.H)
        cls.sampled_hemispheres = n_sampled_hemispheres
        cls.delta_m_meters = data.delta_t
        
        if data.H_format == HFormat.T_Sx_Sy:
            cls.t_max, cls.sensor_width, cls.sensor_height = cls.H.shape
            cls.H = torch.moveaxis(cls.H, source=0, destination=-1)
        else:
            raise RuntimeError(f"Not supported format: {data.H_format}")
        
        min_gt_H, max_gt_H = torch.min(cls.H), torch.max(cls.H)
        cls.H = (cls.H - min_gt_H) / (max_gt_H - min_gt_H)
        
        cls.n_iter = n_iter
    
    @classmethod
    def sample_mkw_light_cone(cls):
        """
        Sample Minkowski Light Cone after illumination events
        """
        assert cls.H is not None, f"Impulse response function is not initialized"
        assert cls.sensor_height is not None and cls.sensor_width is not None
        
        projection_lc = torch.sum(cls.H, dim=-1)
        pdf_illuminated_points = projection_lc / torch.sum(projection_lc)
        pdf_illuminated_points = pdf_illuminated_points.reshape(-1)
        cdf_illuminated_points = torch.cumsum(pdf_illuminated_points, dim=0)
        idxs_samples = torch.searchsorted(cdf_illuminated_points, torch.rand(1)).item()
        
        #TODO: fix this workaround 
        
        width_idx, height_idx = idxs_samples // cls.sensor_width, idxs_samples % cls.sensor_height
        
        return width_idx, height_idx
    
    @classmethod
    def clear(cls):
        cls.n_sampled_hemispheres = None
        cls.t_max = None 
        cls.n_iter = None 
        cls.sensor_width = None 
        cls.sensor_height = None 
        cls.H = None
        cls.delta_m_meters = None
        
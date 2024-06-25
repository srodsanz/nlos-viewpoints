
import torch

from tal.io.capture_data import NLOSCaptureData
from tal.enums import HFormat

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
    def clear(cls):
        """
        Clear context --- set to None so as to reset
        """
        assert cls.read, f"Context must be read so as to clear"
        
        cls.n_sampled_hemispheres = None
        cls.t_max = None 
        cls.n_iter = None 
        cls.sensor_width = None 
        cls.sensor_height = None 
        cls.H = None
        cls.delta_m_meters = None
        
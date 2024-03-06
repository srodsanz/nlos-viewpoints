
import torch

from tal.io.capture_data import NLOSCaptureData
from tal.enums import HFormat


class NeRFContext:
    
    n_sampled_hemispheres = None
    t_max = None 
    n_iter = None # Factor of gradients update
    sensor_width = None
    sensor_height = None
    H = None
    delta_m_meters = None
    
    @classmethod
    def from_ytal(cls, 
                data: NLOSCaptureData,
                n_iter: int,
                n_sampled_hemispheres,
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
            cls.t_max, cls.sensor_width, cls.sensor_height = data.H.shape
            cls.H = torch.moveaxis(cls.H, source=0, destination=-1)
        else:
            raise RuntimeError(f"Not supported format: {data.H_format}")
        
        cls.n_iter = n_iter * cls.t_max
        
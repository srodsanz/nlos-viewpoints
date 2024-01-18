import numpy as np 
import torch

from enum import Enum
from .volume import Volume


class Sampler(Enum):
    """
    Sampling strategies
    """
    SAMPLER_UNIFORM = 0
    SAMPLER_TWO_STAGE = 0


class Renderer:
    """
    Rendering class on volume rendering techniques
    """
    def __init__(self, sampler: Sampler):
        """
        Constructor
        """
        self.sampler = sampler
    
    
    @staticmethod
    def generate_relay_wall(sensor_x, sensor_y, 
                            scale):
        """
        Generate grid-like relay-wall on NLOS scene
        :param sensor_x: width
        :param sensor_y: height
        :param scale: scale of input grid
        """
        assert sensor_x > 0 and sensor_y > 0, f"Resolution on grid components should be positive"
        nx = 2*sensor_x + 1
        ny = 2*sensor_y + 1
        x = np.stack((np.linspace(start=-scale, stop=scale, num=nx),)*ny, axis=1)
        y = np.stack((np.linspace(start=-scale, stop=scale, num=ny),)*nx, axis=0)
        z = np.zeros((nx, ny))
        return np.stack((x, y, z), axis=-1, dtype=np.float32)
    
    @classmethod
    def render_transient(cls, model: torch.nn.Module, 
                         sensor_x, sensor_y, n_t_bins, delta_m_meters, scale):
        """
        Render transient simulation from model output

        Args:
            model (): _description_
            sensor_x (): sensor width
            sensor_y (_type_): sensor height
            n_t_bins (_type_): number of bins
            delta_m_meters (_type_): _description_
        """
        nx = 2 * sensor_x + 1
        ny = 2 * sensor_y + 1
        H = np.zeros((nx, ny, n_t_bins), dtype=np.float32)
        meters_max = n_t_bins * delta_m_meters
        radius_bins = np.linspace(start=0, stop=meters_max, num=n_t_bins) / 2
        az_bins = np.linspace(start=0, stop=np.pi, num=n_t_bins)
        col_bins = np.linspace(start=0, stop=np.pi / 2, num=n_t_bins)
        delta_az = (az_bins[-1] - az_bins[0]) /  n_t_bins
        delta_col = (col_bins[-1] - col_bins[0]) / n_t_bins
        relay_wall = cls.generate_relay_wall(sensor_x=sensor_x, sensor_y=sensor_y, scale=scale)
        
        for i, t_bin in enumerate(n_t_bins):
            radius = radius_bins[i]
            hemisphere_rtf = Volume.sample_2d_hemisphere(radius=radius, n_bins=n_t_bins)
            centers_xyz = relay_wall.reshape(-1, 3)
            spherical_rtf = hemisphere_rtf.reshape(-1, 3)
            centers_reps = np.repeat(centers_xyz, repeats=spherical_rtf.shape[0], axis=0)
            spherical_rtf_reps = np.repeat(spherical_rtf, repeats=centers_xyz.shape[0], axis=0)
            zip_5dfield = np.concatenate((centers_reps, spherical_rtf_reps), axis=1)
            volume_density, albedo = model(zip_5dfield, axis=0)
            H[:, :, i] = ((delta_az * delta_col) / radius ** 2)  * np.sum(np.sin(spherical_rtf[:, 1]) * volume_density * albedo, axis=0)
        
        return H
    
    @classmethod
    def ray_marching(cls, sensor_x, sensor_y, samples_per_ray, scale,
                     furthest_volume):
        """
        Ray marching algorithm for volume rendering on 

        Args:
            sensor_x (int): image width
            sensor_y (int): image height
        """
        origin = np.array([0,0, -1])
        relay_wall = cls.generate_relay_wall(sensor_x=sensor_x, sensor_y=sensor_y, scale=scale)
        raster = np.zeros((sensor_x, sensor_y), dtype=np.float32)
        
        
        
    
        
        
        
        
        
        
        
        
        
    
    
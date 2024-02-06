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
        
    @classmethod
    def transient(x_pos, y_pos, time_bin, delta_m_meters):
        """
        Evaluate transient measurements on given input bins

        Args:
            x_pos (_type_): _description_
            y_pos (_type_): _description_
            time_bin (_type_): _description_
            delta_m_meters (_type_): _description_

        Raises:
            NotImplementedError: _description_

        Returns:
            _type_: _description_
        """
   
    
    @classmethod
    def ray_marching(cls, model, raster_width, raster_height, samples_per_ray, scale,
                     furthest_volume):
        """
        Ray marching algorithm for volume rendering on

        Args:
            sensor_x (int): image width
            sensor_y (int): image height
        """
        raster = np.zeros((raster_width, raster_height), dtype=np.float32)
        relay_wall = cls.generate_relay_wall(sensor_x=raster_width, sensor_y=raster_height, scale=scale)
        p, q, r = (relay_wall[0,0], relay_wall[-1, 0], relay_wall[0, -1])
        normal_uv = np.cross(p - q, p - r)
        normal_unit = normal_uv / np.linalg.norm(normal_uv)
        
        rays_origin = scale * normal_unit
        rays_dirs = relay_wall - rays_origin
        t = np.linspace(start=0, stop=furthest_volume*scale, num=samples_per_ray)
        coefficients_t = np.expand_dims(t, axis=1)
        
        for i in range(raster_width):
            for j in range(raster_height):
                direction = rays_dirs[i, j]
                dir_vectors = np.repeat(np.expand_dims(direction, axis=0), repeats=[samples_per_ray], axis=0)
                points_ray = coefficients_t * dir_vectors
                center = np.repeat(np.expand_dims(relay_wall[i, j], axis=0), repeats=samples_per_ray, axis=0)
                coords_field = np.concatenate((center, points_ray), axis=1)
                volume_density, albedo = model(coords_field, axis=0)
                raster[i, j] = np.max(volume_density * albedo)
        
        return raster
    
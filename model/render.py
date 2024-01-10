import torch

from enum import Enum


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
    def render_transient(colatitude_bins, azimuthal_bins, 
                         delta_colatitude_bins, 
                         delta_azimuthal_bins):
        """
        Render transient measurements by inferred parameters to perform differentiable optimization
        :param colatitude_bins:
        :param azimuthal_bins:
        :param delta_colatitude_bins:
        :param delta_azimuthal_bins:
        """
    

    @staticmethod
    def volume_rendering(voxel_grid, volume_density):
        """
        Volume rendering method 
        """
    


    

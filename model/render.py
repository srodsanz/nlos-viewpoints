import numpy as np 

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
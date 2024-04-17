from enum import Enum

class SphericalFormat(Enum):
    """
    Format of spherical coordinates for implicit conversions

    Args:
        Enum (_type_): _description_

    Returns:
        _type_: _description_
    """
    SF_R_A_C = 0
    SF_R_C_A = 1

class LightFFormat(Enum):
    """
    Format of light field 6d coordinates according to different supported schemes

    Args:
        Enum (_type_): 
        C stands for center coordinate 
        LF is key index
        P stands for point in cartesian coordinates
        S stands for point in spherical coordinates, respect to C

    Returns:
        _type_: 
    """
    LF_X_Y_Z_A_C = 0
    LF_X_Y_Z_C_A = 1
    
    Cartesian_LF = [LF_X_Y_Z_A_C, LF_X_Y_Z_C_A]
    

class LightFCoordinates(Enum):
    """_summary_

    Args:
        Enum (_type_): _description_
    """
    LFC_SPHERICAL = 0
    LFC_CARTESIAN = 1
    
class Sampling(Enum):
    """
    Sampling strategies
    """
    SAMPLING_UNIFORM = 0
    SAMPLING_HIERARCHICAL = 1


class BBox:
    
    def __init__(self, x0, y0, w_offset, h_offset):
        """
        Constructor
        """
        self._x0 = x0
        self._y0 = y0
        self._w_offset = w_offset
        self._h_offset = h_offset
    
    def get_width(self):
        return self._w_offset
    
    def get_height(self):
        return self._h_offset
    
    def to_dict(self):
        """_summary_
        """
        return {
            "x0": self._x0,
            "y0": self._y0,
            "x1": self._x0 + self._w_offset,
            "y1": self._y0 + self._h_offset
        }
    
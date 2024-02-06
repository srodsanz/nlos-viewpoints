from enum import Enum

class Spherical_Format(Enum):
    """
    Format of spherical coordinates for implicit conversions

    Args:
        Enum (_type_): _description_

    Returns:
        _type_: _description_
    """
    SF_R_A_C = 0
    SF_R_C_A = 1

class Cartesian_Format(Enum):
    """
    Format of cartesian coordinates for applied conversions

    Args:
        Enum (_type_): _description_

    Returns:
        _type_: _description_
    """
    X_Y_Z = 0

class LF_Format(Enum):
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
    LF_X_Y_Z_COL_AZ = 0
    LF_X_Y_Z_AZ_COL = 1
    
class V_Format(Enum):
    """
    Format of volume coordinates in grid-like indexing

    Args:
        Enum (_type_): _description_

    Returns:
        _type_: _description_
    """
    V_X_Y_Z_3 = 0
    V_X_3 = 1

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
    




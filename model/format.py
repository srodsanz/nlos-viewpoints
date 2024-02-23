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
    LF_X0_Y0_R_A_C_6 = 0
    LF_X0_Y0_R_C_A_6 = 1

class LightFCoordinates(Enum):
    """_summary_

    Args:
        Enum (_type_): _description_
    """
    LFC_SPHERICAL = 0
    LFC_CARTESIAN = 1
    
class VolSampler(Enum):
    """
    Sampling strategies
    """
    SAMPLER_UNIFORM = 0
    SAMPLER_HIERARCHICAL = 1
    SAMPLER_MANIFOLD = 2
    
class ModelLoss(Enum):
    """_summary_

    Args:
        Enum (_type_): _description_
    """
    LOSS_MSE = 0

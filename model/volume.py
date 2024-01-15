import numpy as np

from typing import Optional
from enum import Enum

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
    LF_Cx_Cy_Cz_P = 0
    LF_P_Cx_Cy_Cz = 1
    LF_Cx_Cy_Cz_S = 2
    LF_S_Cx_Cy_Cz = 3

class V_Format(Enum):
    """
    Format of volume coordinates in grid-like indexing

    Args:
        Enum (_type_): _description_

    Returns:
        _type_: _description_
    """
    V_X_Y_Z_3 = 0
    V_X_3 = 0

class Volume:
    """
    Volume methods for discretization / voxel input grid
    """
    
    @staticmethod
    def generate_volume_xyz(center: np.ndarray, scale,
                            n_sensor_x, n_sensor_y, n_sensor_z):
        """
        Voxelization of given input region by center
        :param center;
        :param scale:
        :param n_sensor_x:
        :param n_sensor_y:
        :param n_sensor_z:
    
        Returns:
            np.ndarray: numpy array of volume points 
        """
        
        nx = 2*n_sensor_x + 1
        ny = 2*n_sensor_y + 1
        nz = 2*n_sensor_z + 1
        x = np.linspace(start=center[0]-scale, stop=center[0]+scale, num=nx)
        y = np.linspace(start=center[1]-scale, stop=center[1]+scale, num=ny)
        z = np.linspace(start=center[2]-scale, stop=center[2]+scale, num=nz)
        X, Y, Z = np.meshgrid(x, y, z)
        coords_xyz = np.stack((X, Y, Z), axis=-1)
        return coords_xyz
    
    @staticmethod
    def sample_2d_hemisphere(center: np.ndarray, radius,
                             n_bins):
        """
        Sample 2D hemispheres on given radius
        :param center:
        :param radius:
        :param n_bins:
        """
        center = np.expand_dims(center, axis=0)
        az_bins = np.linspace(start=0, stop=np.pi, num=n_bins)
        col_bins = np.linspace(start=0, stop=np.pi / 2, num=n_bins)
        r_bins = np.repeat(radius, repeats=n_bins)
        center_bins = np.repeat(center, repeats=n_bins, axis=0)
        R, A, C = np.meshgrid(r_bins, az_bins, col_bins)
        coords_rac = np.stack((R, A, C), axis=-1)
        return coords_rac
 
    @staticmethod
    def spherical2cartesian(center, radius, colatitude, azimuthal):
        """
        Conversion from spherical coordinates to cartesian
        :param center: affine center in xyz coordinates 
        :param radius: radius of sphere
        :param azimuthal: coplanar angle to center plane
        :param colatitude: latitude angle measuring sphere point
        """
        assert radius >= 0, f"Radius {radius} in spherical coordinates must be >= 0"
        assert azimuthal >= 0 and azimuthal < 2*np.pi, f"Azimuthal {azimuthal} must be >=0 and < 2pi"
        assert colatitude >= -np.pi / 2 and colatitude <= np.pi / 2, f"Colatitude {colatitude} must be >= 0 and <= pi"
        x = radius * np.sin(colatitude) * np.cos(azimuthal) + center[0]
        y = radius * np.sin(colatitude) * np.sin(azimuthal) + center[1]
        z = radius * np.cos(colatitude) + center[2]
        return np.concatenate((center, np.array([x, y, z])), axis=1)
    
    
    @staticmethod
    def cartesian2spherical(center, x, y, z):
        """
        Conversion from cartesian to spherical coordinates
        :param center:
        :param x: axis x
        :param y: axis y
        :param z: axis z
        """
        x0 = x - center[0]
        y0 = y - center[1]
        z0 = z - center[2]
        radius = np.sqrt(x0**2 + y0**2 + z0**2)
        colatitude = np.arctan((x0**2 + y0**2) / z0)
        if x0 > 0 and y0 > 0:
            azimuthal = np.arctan(y0 / x0)
        elif x0 > 0 and y0 < 0:
            azimuthal = 2 * np.pi + np.arctan(y0 / x0)
        elif x0 == 0 and y0 > 0:
            azimuthal = np.pi / 2
        elif x0 == 0 and y0 < 0:
            azimuthal = - np.pi / 2
        elif x0 < 0:
            azimuthal = np.pi + np.arctan(y0 / x0)
        else:
            azimuthal = 0
        
        return np.concatenate((center, np.array([radius, colatitude, azimuthal])), axis=1)
    
    
    
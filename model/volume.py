import numpy as np

from typing import Optional

from sympy import numbered_symbols

class Volume:
    """
    Volume methods to voxelize input grid
    """
    
    @staticmethod
    def voxelize(center, n_bins):
        """
        Voxelization of given input region
        :param n_bins: output resolution of the voxel representation
        """
        assert isinstance(center, np.ndarray), f"Center element {center} should be a numpy array"

        dx = 1 / np.sqrt(2)
        dy = 1 / np.sqrt(2)
        dz = 1 / np.sqrt(2)
        x_bins = np.linspace(start=center[0]-center[0]*dx/2, stop=center[0]+center[0]*dx/2, num=n_bins)
        y_bins = np.linspace(start=center[1]-center[1]*dy/2, stop=center[1]+center[1]*dy/2, num=n_bins)
        z_bins = np.linspace(start=center[2]-center[2]*dz/2, stop=center[2]+center[2]*dz/2, num=n_bins)
    
        return np.stack((x_bins, y_bins, z_bins), axis=0)
    
    @staticmethod
    def spherical2cartesian(center, radius, azimuthal, colatitude):
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
        return np.array([x, y, z], dtype=np.float32)
    
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
        
        return np.array([radius, colatitude, azimuthal], dtype=np.float32)
    
    
    @staticmethod
    def sample_2d_hemisphere(
        center: np.ndarray, radius: np.ndarray, n_bins: int
    ):
        """
        Sample points uniformly on a given hemisphere 
        :param center:
        :param radius:
        :param n_bins:
        """
        center = center.reshape(-1, 2)
        colat_bins = np.linspace(start=0, stop=np.pi / 2, num=n_bins)
        azim_bins = np.linspace(start=-np.pi, stop=np.pi, num=n_bins)
        radius_bins = np.repeat(radius, repeats=[n_bins])
        R, C, A = np.meshgrid(radius_bins, colat_bins, azim_bins)
        stacked_coords = np.vstack((R, C, A)).reshape(3, -1).T
        center_reps = np.repeat(center, repeats=[stacked_coords.shape[0]])
        return np.concatenate((stacked_coords, center_reps), axis=1)


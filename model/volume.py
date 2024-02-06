import numpy as np

from .format import Spherical_Format, Cartesian_Format

class Volume:
    """
    Volume methods for discretization / voxel input grid
    """
    
    @staticmethod
    def generate_volume_xyz(center: np.ndarray, scale,
                            sensor_x, sensor_y, sensor_z ):
        """
        Discrete voxelization of given input region by center
        :param center;
        :param scale:
        :param sensor_x:
        :param sensor_y:
        :param sensor_z:
    
        Returns:
            np.ndarray: numpy array of volume points 
        """
        nx = sensor_x
        ny = sensor_y 
        nz = sensor_z
        dx = 1 / np.sqrt(2)
        dy = 1 / np.sqrt(2)
        dz = 1 / np.sqrt(2)
        x = np.linspace(start=center[0]-scale*dx, stop=center[0]+scale*dx, num=nx)
        y = np.linspace(start=center[1]-scale*dy, stop=center[1]+scale*dy, num=ny)
        z = np.linspace(start=center[2]-scale*dz, stop=center[2]+scale*dz, num=nz)
        X, Y, Z = np.meshgrid(x, y, z)
        coords_xyz = np.stack((X, Y, Z), axis=-1)
        return coords_xyz
    
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
        nx = sensor_x
        ny = sensor_y
        x = np.stack((np.linspace(start=-scale, stop=scale, num=nx),)*ny, axis=1)
        y = np.stack((np.linspace(start=-scale, stop=scale, num=ny),)*nx, axis=0)
        z = np.zeros((nx, ny))
        return np.stack((x, y, z), axis=-1, dtype=np.float32)
    
    
    @classmethod
    def sample_2d_hemisphere(cls, radius_max, n_bins, 
            format: Spherical_Format = Spherical_Format.SF_R_A_C):
        """
        Sample 2D hemispheres on given radius and by spherical coordinates

        Args:
            radius_max (float): radius bound for uniform sampling over radius bins
            n_bins (np.ndarray): number of bins for each variable

        Returns:
            _type_: _description_
        """
        assert isinstance(radius_max, float), f"Radius upper bound must be >=0 numeric"
        r_bins = np.linspace(start=0, stop=radius_max, num=n_bins)
        a_bins = np.linspace(start=0, stop=np.pi, num=n_bins)
        c_bins = np.linspace(start=0, stop=np.pi / 2, num=n_bins)
        radius_bins = np.tile(r_bins, reps=n_bins)
        az_bins = np.tile(a_bins, reps=n_bins)
        col_bins = np.tile(c_bins, reps=n_bins)
        R, A, C = np.meshgrid(radius_bins, az_bins, col_bins)
        coords_rac = np.stack((R, A, C), axis=-1) if format == Spherical_Format.SF_R_A_C else np.stack((R, C, A), axis=-1)
        return coords_rac
    
     
    @staticmethod
    def spherical2cartesian(pts, center, format: Spherical_Format = Spherical_Format.SF_R_A_C) -> np.ndarray:
        """
        Convert point from spherical coordinates to cartesian

        Args:
            pts (np.ndarray): cartesian point to be converted
            center (np.ndarray): center point of transient

        Returns:
            _type_: converted points to cartesian coordinates
        """
        assert format in (e for e in Spherical_Format), f"Format of spherical coordinates {format} not yet supported"
        az = pts[:, 1] if format == Spherical_Format.SF_R_A_C else pts[:, 2]
        col = pts[:, 2] if format == Spherical_Format.SF_R_A_C else pts[:, 1]
        r = pts[:, 0]
        assert np.all(r >= 0), f"Radius coordinates must be >= 0"
        assert np.all(az >= 0) and np.all(az <= np.pi), f"Azimuthal {az} must be >= 0 and < 2pi"
        assert np.all(col >= 0) and np.all(col <= np.pi / 2), f"Colatitude {col} must lie in -pi/2, pi/2 interval"
        x = r * np.sin(col) * np.cos(az)
        y = r * np.sin(col) * np.sin(az)
        z = r * np.cos(col)            
        return np.stack((x, y, z), axis=1) + center
     
    
    @classmethod
    def generate_lf_coordinates_xyz(cls, sensor_x, sensor_y, scale, n_precision_bins,
            delta_m_meters, spherical_format: Spherical_Format = Spherical_Format.SF_R_A_C):
        """
        Light field coordinates for input batches
        :param sensor_x: relay wall width
        :param sensor_y: relay wall height
        :param scale: scale factor on grid
        :param n_precision_bins: number of bins for manifold sampling
        :param delta_m_meters: delta discriminant
        """
        relay_wall = cls.generate_relay_wall(sensor_x=sensor_x, sensor_y=sensor_y, scale=scale, n_bins=n_precision_bins)
        radius_max = delta_m_meters * n_precision_bins / 2
        r_hemispheres_rac = cls.sample_2d_hemisphere(radius_max=radius_max, n_bins=n_precision_bins)
        hemispheres = r_hemispheres_rac.reshape(-1, 3)
        centers_xyz = relay_wall.reshape(-1, 3)
        centers_xyz_r = np.tile(centers_xyz, reps=n_precision_bins)
        sampled_cartesian_xyz = cls.spherical2cartesian(hemispheres, centers_xyz_r, format=spherical_format)
        return np.concatenate((sampled_cartesian_xyz, hemispheres), axis=1)
                
        
    
    
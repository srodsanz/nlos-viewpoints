import torch

from .format import SphericalFormat, LightFFormat

class Scene:
    """
    Volume methods for discretization / voxel input grid
    """
    
    def __init__(self, 
            sensor_x, 
            sensor_y,
            scale
        ):
        """
        Constructor
        """
        assert sensor_x > 0 and sensor_y > 0, f"Resolution on grid components should be positive"
        self.sensor_x = sensor_x
        self.sensor_y = sensor_y
        self.scale = scale
    
    def relay_wall(self):
        """
        Generate grid-like relay-wall for NLOS Imaging scene at plane XYO (z = 0)

        Args:
            sensor_x (_type_): sensor width
            sensor_y (_type_): sensor height
            scale (_type_):d scale of proposed grid

        Returns:
            _type_: grid spots
        """
        nx = self.sensor_x
        ny = self.sensor_y
        scale = self.scale
        x = torch.stack((torch.linspace(start=-scale, end=scale, steps=nx),)*ny, dim=1)
        y = torch.stack((torch.linspace(start=-scale, end=scale, steps=ny),)*nx, dim=0)
        z = torch.zeros((nx, ny))
        return torch.stack((x, y, z), dim=-1)
    
    @staticmethod
    def sample_hemispheres(delta_m_meters, n_precision_bins, 
                        n_spherical_bins,
                        format: SphericalFormat = SphericalFormat.SF_R_A_C) -> torch.Tensor:
        """
        Sample 2D hemispheres on given radius and by spherical coordinates

        Args:
            radius_max (_type_): _description_
            n_bins (_type_): _description_
            format (Spherical_Format, optional): _description_. Defaults to Spherical_Format.SF_R_A_C.

        Returns:
            _type_: _description_
        """
        radius_bins = torch.arange(start=0, end=n_precision_bins) * delta_m_meters / 2
        a_bins = torch.linspace(start=0, end=torch.pi, steps=n_spherical_bins)
        c_bins = torch.linspace(start=0, end=torch.pi, steps=n_spherical_bins)
        R, A, C = torch.meshgrid(radius_bins, a_bins, c_bins, indexing="ij")
        if format == SphericalFormat.SF_R_C_A:
            hemispheres = torch.stack((R, C, A), dim=-1)
        else:
            hemispheres = torch.stack((R, A, C), dim=-1)
                
        return hemispheres
    
    @staticmethod
    def spherical2cartesian(light_field, 
                            lf_format: LightFFormat = LightFFormat.LF_X0_Y0_R_A_C_6) -> torch.Tensor:
        """
        Convert points in spherical coordinates to cartesian points

        Args:
            pts (_type_): _description_
            center (_type_): _description_
        Returns:
            torch.Tensor: _description_
        """
        assert light_field.shape[-1] == 6 and light_field.dim() == 6, f"Incorrect shapes of input light field"
        centers = light_field[..., :3]
        hemispheres = light_field[..., 3:]
        
        if lf_format == LightFFormat.LF_X0_Y0_R_A_C_6:
            r, az, col = hemispheres[..., 0], hemispheres[..., 1], hemispheres[..., 2]
        else:
            r, az, col = hemispheres[..., 0], hemispheres[..., 2], hemispheres[..., 1]
        
        assert torch.all(r >= 0), f"Radius bins must be positive"
        assert torch.all(az >= 0) and torch.all(az <= torch.pi), f"Azimuthal bins must be >= 0 and < pi to cover hemisphere"
        assert torch.all(col >= 0) and torch.all(col <= torch.pi), f"Colatitude {col} must lie within -pi / 2, pi / 2 interval"
        
        x = r * torch.sin(col) * torch.cos(az)
        y = r * torch.sin(col) * torch.sin(az)
        z = r * torch.cos(col)
        
        pts = torch.stack((x, y, z), axis=-1) + centers
        cartesian_light_field = torch.cat((pts, hemispheres), axis=-1)
        
        #assert cartesian_light_field.shape[-1] == light_field.shape[-1], f"Incorrect shapes for cartesian light field"
        
        return cartesian_light_field
    
    @staticmethod
    def cartesian2spherical(pts, centers,
                            output_format: SphericalFormat=SphericalFormat.SF_R_A_C):
        """_summary_

        Args:
            pts (_type_): _description_
            centers (_type_): _description_
            output_format (SphericalFormat, optional): _description_. Defaults to SphericalFormat.SF_R_A_C.
            input_format (CartesianFormat, optional): _description_. Defaults to CartesianFormat.CF_X_Y_Z.

        Returns:
            _type_: _description_
        """
        assert pts.shape[-1] == 3 and centers.shape[-1] == 3, \
            f"Incorrect shapes in cartesian coordinates of pts or centers"
        
        x0, y0, z0 = centers[:, 0], centers[:, 1], centers[:, 2]
        x, y, z = pts[:, 0] - x0, pts[:, 1] - y0, pts[:, 2] - z0
        n_pts = pts.shape[0]
        
        r = torch.sqrt(x**2 + y**2 + z**2)
        col = torch.where(r > 0, torch.acos(z / r), torch.zeros(n_pts))
        r_xyo = torch.sqrt(x**2 + y**2)
        mask_y = 2*(y > 0) - 1
        az = torch.where(r_xyo > 0, mask_y * (torch.acos(x / r_xyo)), torch.zeros(n_pts))
        
        if output_format == SphericalFormat.SF_R_A_C:
            stack_pts = torch.stack((r, az, col), dim=-1)
        else:
            stack_pts = torch.stack((r, col, az), dim=-1)
            
        assert stack_pts.shape[-1] == pts.shape[-1], \
            f"Incorrect shapes after coordinate basis change"
        
        return stack_pts
        
    def generate_light_field(self,
            n_hemisphere_bins, n_precision_bins, delta_m_meters, 
            spherical_format:SphericalFormat=SphericalFormat.SF_R_A_C,
            lf_format: LightFFormat = LightFFormat.LF_X0_Y0_R_A_C_6
    ):
        """
        Generate light-field coordinates

        Args:
            n_hemisphere_bins (_type_): _description_
            n_precision_bins (_type_): _description_
            delta_m_meters (_type_): _description_
        """
        relay_wall = self.relay_wall()
        hemispheres = self.sample_hemispheres(delta_m_meters=delta_m_meters, n_precision_bins=n_precision_bins, 
                n_spherical_bins=n_hemisphere_bins, format=spherical_format)
        
        assert relay_wall.shape[-1] == 3, f"Relay wall contains xyz coordinates"
        assert hemispheres.shape[-1] == 3, f"Hemispheres coordinates does not contain spherical samplings"
        
        hem_ext = hemispheres[None, None, ...].expand(*relay_wall.shape[:2], *hemispheres.shape)
        rw_ext = relay_wall[:, :, None, None, None, :].expand((*relay_wall.shape[:2], *hemispheres.shape[:3], relay_wall.shape[-1]))
        
        if lf_format == LightFFormat.LF_X0_Y0_R_A_C_6:
            lf = torch.cat((rw_ext, hem_ext), dim=-1)
        
        else:
            hem_ext = torch.moveaxis(hem_ext, source=-2, destination=-1)
            lf = torch.cat((rw_ext, hem_ext), dim=-1)

        cartesian_light_field = self.spherical2cartesian(lf, lf_format=lf_format)
        
        return cartesian_light_field
        
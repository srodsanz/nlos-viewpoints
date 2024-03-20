import torch

from .format import SphericalFormat, LightFFormat

class Scene:
    """
    Volume methods for discretization / voxel input grid
    """
    
    def __init__(self, 
            sensor_x, 
            sensor_y,
            scale=1
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
        x = torch.stack((torch.linspace(start=-scale, end=scale, steps=2*nx+1)[1::2],)*ny, dim=1)
        y = torch.stack((torch.linspace(start=-scale, end=scale, steps=2*ny+1)[1::2],)*nx, dim=0)
        z = torch.zeros((nx, ny))
        return torch.stack((x, y, z), dim=-1).type(torch.float32)
    
    def sample_pdf_hemispheres(self,
                    output_map,
                    delta_m_meters,
                    time_start,
                    time_end,
                    n_spherical_fine_bins,
                    n_spherical_coarse_bins,
                    input_format: SphericalFormat = SphericalFormat.SF_R_A_C,
                    sampling_format: SphericalFormat = SphericalFormat.SF_R_A_C
        ) -> torch.Tensor:
        """
        Importance sampling of different PDFs related to geometry estimation

        Args:
            delta_m_meters (_type_): _description_
            time_start (_type_): _description_
            time_end (_type_): _description_
            n_spherical_fine_bins (_type_): _description_
            sampling_format (SphericalFormat, optional): _description_. Defaults to SphericalFormat.SF_R_A_C.

        Returns:
            torch.Tensor: _description_
        """
        assert output_map.dim() == 4, f"Incorrect shapes for input dimensions"
        assert n_spherical_fine_bins <= n_spherical_coarse_bins, f"Fine sampling needs to be performed over less elements"
        
        radius_bins = torch.arange(start=time_start, end=time_end) * delta_m_meters / 2
        az_coarse_bins = torch.arange(start=0, end=torch.pi, steps=n_spherical_coarse_bins)
        col_coarse_bins = torch.arange(start=0, end=torch.pi, steps=n_spherical_coarse_bins)
        
        if input_format == SphericalFormat.SF_R_A_C:
            pdf_az, pdf_col = torch.sum(output_map, axis=-1), torch.sum(output_map, axis=-2)        
        else:
            pdf_col, pdf_az = torch.sum(output_map, axis=-1), torch.sum(output_map, axis=-2)
        
        pdf_az, pdf_col = pdf_az / torch.sum(pdf_az, axis=-1)[..., None].expand(*pdf_az.shape), pdf_col / torch.sum(pdf_col, axis=-1)[..., None].expand(*pdf_col.shape)
        assert pdf_az.shape[0] == pdf_col.shape[0] and pdf_az.shape[-1] == pdf_col.shape[-1], f"Shapes on PDFs should match"
        
        cdf_az, cdf_col = torch.cumsum(pdf_az, axis=-1), torch.cumsum(pdf_col, axis=-1)
        sampled_uniforms = torch.rand((radius_bins.shape[0], n_spherical_fine_bins))
        idxs_az = torch.searchsorted(cdf_az, sampled_uniforms)
        idxs_col = torch.searchsorted(cdf_col, sampled_uniforms)
        az_bins = az_coarse_bins[idxs_az]
        col_bins = col_coarse_bins[idxs_col]
        
        R, A, C = radius_bins[..., None, None].extend(radius_bins.shape[0], n_spherical_fine_bins, n_spherical_fine_bins), \
            az_bins[..., None].extend(*az_bins.shape, n_spherical_fine_bins), col_bins[..., None].extend(*col_bins.shape, n_spherical_fine_bins)
        
        if sampling_format == SphericalFormat.SF_R_A_C:
            hemispheres = torch.stack((R, A, C), axis=-1)
        else:
            hemispheres = torch.stack((R, C, A), axis=-1)
        
        return hemispheres
        
    def sample_uniform_hemispheres(
                        self,
                        delta_m_meters,
                        arg_start,
                        arg_end,
                        time_start,
                        time_end,
                        n_spherical_coarse_bins,
                        sampling_format:SphericalFormat
        ) -> torch.Tensor:
        """
        Sample 2D hemispheres on given radius and by spherical coordinates

        Args:
            radius_max (_type_): _description_
            n_bins (_type_): _description_
            format (Spherical_Format, optional): _description_. Defaults to Spherical_Format.SF_R_A_C.

        Returns:
            _type_: _description_
        """
        radius_bins = torch.arange(start=time_start, end=time_end) * delta_m_meters / 2
        a_bins = torch.linspace(start=arg_start, end=arg_end, steps=n_spherical_coarse_bins)
        c_bins = torch.linspace(start=arg_start, end=arg_end, steps=n_spherical_coarse_bins)
        R, A, C = torch.meshgrid(radius_bins, a_bins, c_bins, indexing="ij")
    
        if sampling_format == SphericalFormat.SF_R_C_A:
            hemispheres = torch.stack((R, C, A), dim=-1)
        else:
            hemispheres = torch.stack((R, A, C), dim=-1)
        
        return hemispheres
    
    
    def spherical2cartesian(self, 
                            spherical_light_field, 
                            spherical_format
        ) -> torch.Tensor:
        """
        Convert points in spherical coordinates to cartesian points

        Args:
            pts (_type_): _description_
            center (_type_): _description_
        Returns:
            torch.Tensor: _description_
        """
        assert spherical_light_field.shape[-1] == 6 and spherical_light_field.dim() == 6, f"Incorrect shapes of input light field"
        assert spherical_format in [SphericalFormat.SF_R_A_C, SphericalFormat.SF_R_C_A], f"Incorrect input LF format"
        
        centers = spherical_light_field[..., :3]
        hemispheres = spherical_light_field[..., 3:]
        
        if spherical_format == SphericalFormat.SF_R_A_C:
            r, az, col = hemispheres[..., 0], hemispheres[..., 1], hemispheres[..., 2]
        else:
            r, az, col = hemispheres[..., 0], hemispheres[..., 2], hemispheres[..., 1]
        
        assert torch.all(r >= 0), f"Radius bins must be positive"
        assert torch.all(az >= 0) and torch.all(az <= torch.pi), f"Azimuthal bins must be >= 0 and < pi to cover hemisphere"
        assert torch.all(col >= 0) and torch.all(col <= torch.pi), f"Colatitude {col} must lie within -pi / 2, pi / 2 interval"
        
        x = r * torch.sin(col) * torch.cos(az)
        y = r * torch.cos(col)
        z = r * torch.sin(col) * torch.sin(az)
        
        pts = torch.stack((x, y, z), axis=-1) + centers
        
        cartesian_light_field = torch.cat((pts, hemispheres), axis=-1)
        assert cartesian_light_field.shape[-1] == spherical_light_field.shape[-1], f"Incorrect shapes for cartesian light field after conversion change"
        
        return cartesian_light_field
    
    def cartesian2spherical(self, 
                            cartesian_light_field,
                            centers
        ):
        """_summary_

        Args:
            pts (_type_): _description_
            centers (_type_): _description_
            output_format (SphericalFormat, optional): _description_. Defaults to SphericalFormat.SF_R_A_C.
            input_format (CartesianFormat, optional): _description_. Defaults to CartesianFormat.CF_X_Y_Z.

        Returns:
            _type_: _description_
        """
        assert centers.shape[-1] == 3 and cartesian_light_field.shape[-1] == 6 and cartesian_light_field.dim() == centers.dim(), f"Centers and light field must have same dims"
        
        lf_h = cartesian_light_field[..., 3:]
        spherical_light_field = torch.cat((centers, lf_h))
        
        return spherical_light_field
        
    
    def generate_light_field(self,
            arg_start,
            arg_end,
            time_start,
            time_end,
            n_spherical_coarse_bins, 
            delta_m_meters,
            spherical_format:SphericalFormat=SphericalFormat.SF_R_A_C,
            output_lf_format:LightFFormat=LightFFormat.LF_X_Y_Z_A_C,
    ):
        """
        Generate light-field coordinates in spherical and cartesian coordinates. Both include viewing direction

        Args:
            n_hemisphere_bins (_type_): _description_
            n_precision_bins (_type_): _description_
            delta_m_meters (_type_): _description_
        """
        assert output_lf_format in [LightFFormat.LF_X_Y_Z_A_C, LightFFormat.LF_X_Y_Z_C_A], f"Incorrect light field formats for batch"
        
        relay_wall = self.relay_wall()

        hemispheres = self.sample_uniform_hemispheres(
                delta_m_meters=delta_m_meters, 
                time_start=time_start,
                time_end=time_end,
                arg_start=arg_start,
                arg_end=arg_end,
                n_spherical_coarse_bins=n_spherical_coarse_bins, 
                sampling_format=spherical_format
        )
        
        assert relay_wall.shape[-1] == 3, f"Relay wall contains xyz coordinates"
        assert hemispheres.shape[-1] == 3, f"Hemispheres coordinates does not contain spherical samplings"
        
        hem_ext = hemispheres[None, None, ...].expand(*relay_wall.shape[:2], *hemispheres.shape)
        rw_ext = relay_wall[:, :, None, None, None, :].expand((*relay_wall.shape[:2], *hemispheres.shape[:3], relay_wall.shape[-1]))
        
        if output_lf_format == LightFFormat.LF_X_Y_Z_A_C:
            spherical_light_field = torch.cat((rw_ext, hem_ext), dim=-1)
        
        else:
            hem_ext = torch.moveaxis(hem_ext, source=-2, destination=-1)
            spherical_light_field = torch.cat((rw_ext, hem_ext), dim=-1)

        cartesian_light_field = self.spherical2cartesian(spherical_light_field=spherical_light_field, 
                                                        spherical_format=spherical_format)
        
        return cartesian_light_field
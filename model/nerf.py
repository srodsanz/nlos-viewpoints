import torch

from torch import nn

from .format import LightFFormat

class NLOSNeRF(nn.Module):
    """
    NLOS imaging addressed by NeRF functions. Standalone architecture 
    """
        
    def __init__(self, 
                positional_encoding=True, 
                n_input_position=3,
                n_input_views=2, 
                n_outputs=256, 
                n_hidden_layers=8, 
                skips=[4],
                length_embeddings=5
    ):
        """
        Constructor
        :param n_input_components: number of input components
        :param n_outputs: number of outputs
        :param n_hidden_layers: number of hidden units for used network
        :param device: architecture device URI
        """
        
        super(NLOSNeRF, self).__init__()
        
        if positional_encoding:
            input_nn_pts = n_input_position + 2 * n_input_position * length_embeddings
            input_nn_views = n_input_views + 2 * n_input_views * length_embeddings
            self.length_embeddings = length_embeddings        
        
        else:
            input_nn_pts = n_input_position
            input_nn_views = n_input_views
        
        self.n_input_position = input_nn_pts
        self.n_input_views = input_nn_views
        self.skips = skips
        self.n_outputs = n_outputs
        self.n_hidden_layers = n_hidden_layers
        self.hidden_components = torch.nn.ModuleList(
            [torch.nn.Linear(self.n_input_position, n_outputs)] + [torch.nn.Linear(n_outputs, n_outputs) if i not in skips else torch.nn.Linear(n_outputs + input_nn_pts, n_outputs)
                for i in range(self.n_hidden_layers)]
        )
        self.input_views = torch.nn.ModuleList([torch.nn.Linear(input_nn_views + n_outputs, n_outputs // 2)])
        self.final_linear = torch.nn.Linear(n_outputs, n_outputs)
        self.volume_density_layer = torch.nn.Linear(n_outputs, 1)
        self.albedo_output = torch.nn.Linear(n_outputs // 2, 1)

    
    def forward(self, x):
        """
        Forward model for input architecture on the 
        :param x: input for hidden layer
        """
        
        input_pts, input_views = torch.split(x, [self.n_input_position, self.n_input_views], dim=-1)
        h = input_pts

        for i, _ in enumerate(self.hidden_components):
            h = self.hidden_components[i](h)
            h = torch.nn.functional.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], dim=-1)
        
        # Output cycle
        volume_density = self.volume_density_layer(h)
        volume_density = torch.abs(volume_density)
        feature = self.final_linear(h)
        h = torch.cat((feature, input_views), dim=-1)
        
        for i, _ in enumerate(self.input_views):
            h = self.input_views[i](h)
            h = torch.nn.functional.relu(h)
        
        albedo = self.albedo_output(h)
        albedo = torch.abs(albedo)
        
        return torch.cat((volume_density, albedo), dim=-1)
    
    def fourier_encoding(self, 
            in_: torch.Tensor,
            x_min, x_max, y_min, y_max, z_min, z_max,
            lf_format: LightFFormat=LightFFormat.LF_X_Y_Z_A_C
    ):
        """
        Positional encoding computation from NDC coordinates

        Args:
            in_ (_type_): _description_
        """
        assert in_.shape[-1] == 5, f"Incorrect dimensions for input in positional encoding"
        assert hasattr(self, "length_embeddings"), f"Current object does not contain length embedding"
        
        length = self.length_embeddings
        
        basis = (2 ** torch.arange(length)) * torch.pi
        
        x, y, z = in_[..., 0, None], in_[..., 1, None], in_[..., 2, None]
        
        if lf_format == LightFFormat.LF_X_Y_Z_A_C:
            az, col = in_[..., -2, None], in_[..., -1, None]
        else:
            az, col = in_[..., -1, None], in_[..., -2, None]
        
        x = 2 * ((x - x_min) / (x_max - x_min)) - 1
        y = 2 * ((y - y_min) / (y_max - y_min)) - 1 
        z = 2 * ((z - z_min) / (z_max - z_min)) - 1 
        az = 2 * (az / torch.pi) - 1
        col = 2 * (col / torch.pi) - 1
        
        x_f = x * basis
        y_f = y * basis
        z_f = z * basis
        az_f = az * basis
        col_f = col * basis
        
        x_sin, x_cos = torch.sin(x_f), torch.cos(x_f)
        y_sin, y_cos = torch.sin(y_f), torch.cos(y_f)
        z_sin, z_cos = torch.sin(z_f), torch.cos(z_f)
        az_sin, az_cos = torch.sin(az_f), torch.cos(az_f)
        col_sin, col_cos = torch.sin(col_f), torch.cos(col_f)
        
        if lf_format == LightFFormat.LF_X_Y_Z_A_C:
            outputs = torch.cat((x, x_sin, x_cos, y, y_sin, y_cos, z, z_sin, z_cos, 
                                az, az_sin, az_cos, col, col_sin, col_cos), dim=-1)
        else:
            outputs = torch.cat((x, x_sin, x_cos, y, y_sin, y_cos, z, z_sin, z_cos, 
                                col, col_sin, col_cos, az, az_sin, az_cos), dim=-1)

        return outputs
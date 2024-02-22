import torch

from torch import nn

class NLOSNeRF(nn.Module):
    """
    NLOS imaging addressed by NeRF functions. Standalone architecture 
    """
        
    def __init__(self, device, n_input_position=3,
                n_input_views=3, n_outputs=256, n_hidden_layers=6, skip_idx=5,
                length_embeddings=5):
        """
        Constructor
        :param n_input_components: number of input components
        :param n_outputs: number of outputs
        :param n_hidden_layers: number of hidden units for used network
        :param device: architecture device URI
        """
        assert (device.startswith("cuda") and torch.cuda.is_available()) or (device == "cpu"), \
        f"The specified device: {device} is not available or unknown"

        self.device = device
        self.n_input_position = 2 * n_input_position * length_embeddings
        self.n_input_views = 2 * n_input_views * length_embeddings
        self.skip_idx = skip_idx
        self.length_embeddings = length_embeddings
        self.n_outputs = n_outputs
        self.n_hidden_layers = n_hidden_layers
        self.hidden_components = torch.nn.ModuleList(
            [torch.nn.Linear(self.n_input_position, n_outputs)] + [torch.nn.Linear(n_outputs, n_outputs) for _ in range(self.n_hidden_layers)]
        )
        self.input_views = torch.nn.ModuleList([torch.nn.Linear(n_input_views + n_outputs, n_outputs // 2)])
        self.final_linear = torch.nn.Linear(n_outputs, n_outputs)
        self.volume_density_layer = torch.nn.Linear(n_outputs, 1)
        self.albedo_output = torch.nn.Linear(n_outputs // 2, 1)

    
    def forward(self, x):
        """
        Forward model for input architecture on the 
        :param x: input for hidden layer
        """
        def positional_encoding(self, 
                            in_: torch.Tensor):
            """
            Positional encoding computation

            Args:
                in_ (_type_): _description_
            """
            assert in_.shape[-1] == 6 and in_.dim() == 2, f"Incorrect dimensions for input in positional encoding"
            
            length = self.length_embeddings
            fourier_basis = 2 ** torch.arange(length)
            terms = in_.unsqueeze(-1) * fourier_basis.unsqueeze(-2)
            pos_encoding = torch.cat((torch.cos(terms), torch.sin(terms)), dim=-1).reshape(*in_.shape, -1)
            return pos_encoding
        
        x = positional_encoding(x)
        input_pts, input_views = torch.split(x, [self.n_input_position, self.n_input_views])

        for i, _ in enumerate(self.hidden_components):
            h = self.hidden_components[i](h)
            h = torch.nn.functional.relu(h)
            if i in self.skip_idx:
                h = torch.cat([input_pts, h], dim=-1)
        
        # Output cycle
        
        volume_density = self.volume_density_layer(h)
        volume_density = torch.sigmoid(volume_density)
        feature = self.final_linear(h)
        h = torch.cat((feature, input_views), dim=-1)
        for i, _ in enumerate(self.input_views):
            h = self.input_views[i](h)
            h = torch.nn.functional.relu(h)
        
        albedo = self.albedo_output(h)
        albedo = torch.sigmoid(albedo)
        
        return torch.cat((volume_density, albedo), dim=1)
    
    

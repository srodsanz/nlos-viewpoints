import torch

from torch import nn

class NLOSNeRF(nn.Module):
    """
    NLOS imaging addressed by NeRF functions. Standalone architecture 
    """
        
    def __init__(self, n_input_position=3,
                 n_input_views=3, n_outputs=256, n_hidden_layers=9, skip_idx=8, device="cpu",
                 length_embeddings=10):
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
        self.n_input_position = n_input_position
        self.n_input_views = n_input_views
        self.skip_idx = skip_idx
        self.length_embeddings = length_embeddings
        self.n_outputs = n_outputs
        self.n_hidden_layers = n_hidden_layers
        self.hidden_components = torch.nn.ModuleList(
            [torch.nn.Linear(self.n_input_position * self.length_embeddings, n_outputs)] + [torch.nn.Linear(n_outputs, n_outputs) for _ in range(self.n_hidden_layers)]
        )
        self.input_views = torch.nn.ModuleList([torch.nn.Linear(n_input_position * self.length_embeddings + n_outputs, n_outputs // 2)])
        self.final_linear = torch.nn.Linear(n_outputs, n_outputs)
        self.volume_density_layer = torch.nn.Linear(n_outputs, 1)
        self.albedo_output = torch.nn.Linear(n_outputs // 2, 1)

    
    def forward(self, x):
        """
        Forward model for input architecture on the 
        :param x: input for hidden layer
        """

        def positional_encoding(feature, length_embeddings=self.length_embeddings):
            """
            Compute positional encodings using Fourier transform for several frequencies
            :param feature (torch.Tensor): input tensor to compute feature vectors
            :param length_embeddings (int): length of computed embeddings
            """
            assert length_embeddings >= 1, f"Length of embeddings {length_embeddings} must be >= 1"
            assert isinstance(feature, torch.Tensor), f"input feature should be cast to tensor"
            encodings = []
            for i in range(-length_embeddings // 2, length_embeddings // 2):
                encodings.append(torch.sin(2. ** i * feature))
                encodings.append(torch.cos(2. ** i * feature))
            
            return torch.stack(encodings, dim=0)


        input_pts, input_views = torch.split(x, [self.n_input_position, self.n_input_views])
        h = positional_encoding(input_pts) # Variable for forwarding

        for i, layer in enumerate(self.hidden_components):
            h = self.hidden_components[i](h)
            h = torch.nn.functional.relu(h)
            if i in self.skip_idx:
                h = torch.cat([input_pts, h], dim=-1)
        
        # Output cycle
        
        volume_density = self.volume_density_layer(h)
        volume_density = torch.sigmoid(volume_density)
        feature = self.final_linear(h)
        h = torch.cat((feature, positional_encoding(input_views)), dim=-1)
        for i, _ in enumerate(self.input_views):
            h = self.input_views[i](h)
            h = torch.nn.functional.relu(h)
        
        albedo = self.albedo_output(h)
        albedo = torch.sigmoid(albedo)
        
        return volume_density, albedo

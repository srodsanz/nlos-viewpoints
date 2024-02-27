import torch

from torch import nn

class NLOSNeRF(nn.Module):
    """
    NLOS imaging addressed by NeRF functions. Standalone architecture 
    """
        
    def __init__(self, n_input_position=3,
                n_input_views=3, n_outputs=256, n_hidden_layers=8, skips=[4],
                length_embeddings=10):
        """
        Constructor
        :param n_input_components: number of input components
        :param n_outputs: number of outputs
        :param n_hidden_layers: number of hidden units for used network
        :param device: architecture device URI
        """
        
        super(NLOSNeRF, self).__init__()
        
        input_nn_pts = 2 * n_input_position * length_embeddings
        input_nn_views = 2 * n_input_views * length_embeddings
        
        self.n_input_position = input_nn_pts
        self.n_input_views = input_nn_views
        self.skips = skips
        self.length_embeddings = length_embeddings
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

        for i, layer in enumerate(self.hidden_components):
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
    
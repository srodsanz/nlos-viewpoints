import torch

def positional_encoding(self, in_: torch.Tensor):
        """
        Positional encoding computation

        Args:
            in_ (_type_): _description_
        """
        assert in_.shape[-1] == 6, f"Incorrect dimensions for input in positional encoding"
        
        length = self.length_embeddings
        fourier_basis = 2 ** torch.arange(length)
        terms = in_.unsqueeze(-1) * fourier_basis.unsqueeze(-2)
        pos_encoding = torch.cat((torch.cos(terms), torch.sin(terms)), dim=-1).reshape((*in_.shape[:-1], -1))
        return pos_encoding

    

import torch

def translation_transform(t, device):
    """
    Translation matrix

    Args:
        t (_type_): _description_
    """
    t_mat = torch.Tensor([[1, 0, 0, 0], 
                            [0, 1, 0, 0],
                            [0, 0, 1, t],
                            [0, 0, 0, 1]]).to(device=device)
    
    return t_mat

def rotation_oyz(theta, device):
    """
    Rotation over plane OYZ

    Args:
        theta (_type_): _description_
        device (_type_): _description_
    """
    rot_mat = torch.Tensor([[1, 0, 0, 0],
                            [0, torch.cos(theta), -torch.sin(theta), 0, 0],
                            [0, torch.sin(theta), torch.cos(theta), 0, 0],
                            [0, 0, 0, 1]]).to(device=device)
    return rot_mat

def rotation_xoz(theta, device):
    """
    Rotation over plane XOZ

    Args:
        theta (_type_): _description_
        device (_type_): _description_
    """
    rot_mat = torch.Tensor([[torch.cos(theta), 0, -torch.sin(theta), 0],
                            [0, 1, 0, 0]
                            [torch.sin(theta), 0, torch.cos(theta), 0],
                            [0, 0, 0, 1]]).to(device=device)
    return rot_mat

def rotation_xyo(theta, device):
    """
    Rotation over plane XYO

    Args:
        theta (_type_): _description_
        device (_type_): _description_
    """
    rot_mat = torch.Tensor([[torch.cos(theta), -torch.sin(theta), 0, 0],
                            [torch.sin(theta), torch.cos(theta), 0, 0]
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]]).to(device=device)
    return rot_mat

def transform(t, theta, plane: str, 
                device:str):
    """_summary_

    Args:
        t (_type_): _description_
        theta (_type_): _description_
        plane (str): _description_
        device (str): _description_
    """
    planes = ["xy", "xz", "yz"]
    assert plane in planes, f"Not supported rotation for different planes. Allowed are {planes}"
    
    t_mat = translation_transform(t, device=device)
    
    if plane == "xy":
        rot_mat = rotation_xyo(theta, device=device)
    
    elif plane == "xz":
        rot_mat = rotation_xoz(theta, device=device)
    
    else:
        rot_mat = rotation_oyz(theta, device=device)
    
    return rot_mat * t_mat
    
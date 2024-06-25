import torch

from enum import Enum

class NeTFLoss(Enum):
    """
    Class to implement different losses used in the NeTF model
    """
    
    LOSS_MSE = 0
    LOSS_SE = 1
    
    @classmethod
    def func_from_id(cls, id):
        """_summary_

        Args:
            id (_type_): _description_

        Returns:
            _type_: _description_
        """
        if id == cls.LOSS_MSE.name:
            return cls.mean_squared_error
        
        else:
            return cls.squared_error
        

    @staticmethod
    def squared_error(
        transient_pred: torch.Tensor,
        transient_gt: torch.Tensor
    ):
        """
        Error based on mean squared error
        :param transient_pred: transient prediction over position / time bin
        :param transient_gt: transient ground truth to compare with
        """
        return torch.sum(
            torch.square(transient_gt - transient_pred)
        )
    
    @staticmethod
    def mean_squared_error(
        transient_pred: torch.Tensor,
        transient_gt: torch.Tensor
    ) -> torch.Tensor:
        
        """_summary_

        Args:
            transient_pred (torch.Tensor): _description_
            transient_gt (torch.Tensor): _description_
        """
        return torch.mean(
            torch.square(transient_gt - transient_pred)
        )
    
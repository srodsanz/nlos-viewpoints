import torch

from typing import Optional
from .format import ModelLoss

class NeTFLoss:
    """
    Class to implement different losses used in the NeTF model
    """

    @classmethod
    def from_id(cls, id,
                transient_pred: torch.Tensor,
                transient_gt: torch.Tensor
        ) -> Optional[torch.Tensor]:
        """
        Get loss value from different IDs defined for the used implementation
        :param transient_pred: transient measurements predicted by renderer
        :param transient_gt: ground truth transient measurement
        """
        loss = None
        if id == ModelLoss.LOSS_MSE:
            loss =  cls.mean_squared_error(transient_pred=transient_pred, transient_gt=transient_gt)
        
        return loss
        

    @staticmethod
    def mean_squared_error(
        transient_pred: torch.Tensor,
        transient_gt: torch.Tensor
    ):
        """
        Error based on mean squared error
        :param transient_pred: transient prediction over position / time bin
        :param transient_gt: transient ground truth to compare with
        """
        return torch.mean(
            torch.square(transient_gt - transient_pred)
        )
    
    
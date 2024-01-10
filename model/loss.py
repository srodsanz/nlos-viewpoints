import torch

from torch.nn import KLDivLoss
from enum import Enum

class NeTFLoss(Enum):
    """
    Class to implement different losses used in the NeTF model
    """
    LOSS_MSE = 0

    @staticmethod
    def from_id(id,
                transient_pred: torch.Tensor,
                transient_gt: torch.Tensor
        ):
        """
        Get loss value from different IDs defined for the used implementation
        :param transient_pred:
        :param transient_gt:
        """
        raise NotImplementedError("NYI - Function not yet implemented")

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
        factor_sq = (transient_gt - transient_pred) ** 2
        return torch.mean(factor_sq)
    
    
import h5py
import os
import numpy as np 

from abc import ABC

class LazyDataset(ABC):
    """
    Base class of implementation in Lazily-evaluated dataset

    Args:
        ABC (_type_): _description_
    """
    def __init__(self):
        self.is_read = False
        self.data = None
    
    def read(self, path):
        """_summary_

        Args:
            path (_type_): _description_
        """
        assert os.path.exists(path), f"Path {path} does not exist in Filesystem"
        raise NotImplementedError("Not implemented for generic lazy datasets")


class YTALDataset(LazyDataset):
    """
    Read y-tal framework resulting dataset

    Args:
        ABC (_type_): _description_
    """
    #TODO: Implement by inheritance of interface

class ZNLOSDataset(LazyDataset):
    """
    ZNLOS simulations synthetic dataset 

    Args:
        LazyDataset (_type_): _description_
    """
    #TODO: implement by inheritance of interface

class FKMigrationDataset(LazyDataset):
    """_summary_

    Args:
        LazyDataset (_type_): _description_
    """
    #TODO: Implement by inheritance of interface
        
        
    

import torch
import cv2
import numpy as np

from tal.io.capture_data import NLOSCaptureData
from tal.enums import HFormat

from .format import BBox


class NeRFContext:
    
    n_sampled_hemispheres = None
    t_max = None 
    n_iter = None
    sensor_width = None
    sensor_height = None
    H = None
    delta_m_meters = None
    
    @classmethod
    def from_ytal(cls, 
                data: NLOSCaptureData,
                n_iter: int,
                n_sampled_hemispheres
        ):
        """
        Import NeRF Context from simulated transient

        Args:
            data (NLOSCaptureData): _description_
        """
        
        cls.H = torch.from_numpy(data.H)
        cls.sampled_hemispheres = n_sampled_hemispheres
        cls.delta_m_meters = data.delta_t
        
        if data.H_format == HFormat.T_Sx_Sy:
            cls.t_max, cls.sensor_width, cls.sensor_height = cls.H.shape
            cls.H = torch.moveaxis(cls.H, source=0, destination=-1)
        else:
            raise RuntimeError(f"Not supported format: {data.H_format}")
        
        min_gt_H, max_gt_H = torch.min(cls.H), torch.max(cls.H)
        cls.H = (cls.H - min_gt_H) / (max_gt_H - min_gt_H)
        
        cls.n_iter = n_iter
    
    @classmethod
    def estimate_mkw_geometry_bbox(cls,
                                n_clusters=2,
                                max_iter=1000, eps=1e-5):
        """
        Apply unsupervised clustering algorithm to estimate the projection of hidden geometry in the relay wall
        This operator acts as an unsupervised binarization in floating point data, avoiding direct thresholding
        
        @pre --- NeRF context must be initialized
        """
        assert cls.H is not None, f"Impulse response function is not initialized"
        assert cls.sensor_height is not None and cls.sensor_width is not None, f"Transient measurement not initialized"
        
        projection_lc_mkw = torch.sum(cls.H, dim=-1).numpy()
        criteria = (cv2.TermCriteria_EPS + cv2.TERM_CRITERIA_MAX_ITER, max_iter, eps)
        
        _, labels, centers = cv2.kmeans(projection_lc_mkw, 
                                        K=n_clusters,
                                        bestLabels=None, 
                                        criteria=criteria,
                                        attempts=max_iter,
                                        centers=cv2.KMEANS_RANDOM_CENTERS)
        
        centers = np.uint8(centers)
        clusters = centers[labels.flatten()]
                
        centers = np.uint8(centers)
        clusters = centers[labels.flatten()]
        bbox_idxs_x, bbox_idxs_y = np.where(clusters == 1)
        width_min = np.min(bbox_idxs_x)
        height_min = np.min(bbox_idxs_y)
        w_offset = np.max(bbox_idxs_x) - width_min
        h_offset = np.max(bbox_idxs_y) - height_min
        

        return BBox(x0=width_min, 
                    y0=height_min,
                    w_offset=w_offset,
                    h_offset=h_offset
        )
    
    @classmethod
    def clear(cls):
        """
        Clear context --- set to None in order to GC
        """
        cls.n_sampled_hemispheres = None
        cls.t_max = None 
        cls.n_iter = None 
        cls.sensor_width = None 
        cls.sensor_height = None 
        cls.H = None
        cls.delta_m_meters = None
        
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import MinkowskiEngine as ME

def ic_collate_fn(data_labels):
    coords, feats, labels = list(zip(*data_labels))

    # Create batched coordinates for the SparseTensor input
    bcoords = ME.utils.batched_coordinates(coords)

    # Concatenate all lists
    feats_batch = torch.from_numpy(np.concatenate(feats, 0)).float()
    labels_batch = torch.from_numpy(np.concatenate(labels, 0)).float()
    
    return bcoords, feats_batch, labels_batch

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def LogCoshLoss(pred, truth):
    x = pred - truth
    return (x + torch.nn.functional.softplus(-2.0 * x) - np.log(2.0)).mean()

def CosineAngleLoss(pred, truth):
    x = torch.mean(torch.abs(F.cosine_similarity(truth, pred)))
    return 1. - x
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import MinkowskiEngine as ME
import matplotlib.pyplot as plt

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
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0+1e-7, 1.0-1e-7))

def LogCoshLoss(pred, truth):
    x = pred - truth
    return (x + torch.nn.functional.softplus(-2.0 * x) - np.log(2.0)).mean()

def CosineSimilarityLoss(pred, truth):
    x = F.cosine_similarity(pred, truth).mean()
    return 1. - x

def AngularDistanceLoss(pred, truth, weights=None, eps=1e-7, reduction="mean"):
    if weights is None:
        x = torch.acos(torch.clamp(F.cosine_similarity(pred, truth), min=-1.+eps, max=1.-eps)) / np.pi
    else:
        x = (torch.acos(torch.clamp(F.cosine_similarity(pred, truth), min=-1.+eps, max=1.-eps)) / np.pi) * weights
    if reduction == "mean":
        return x.mean()
    else:
        return x

def AngularDistanceLossV2(pred, truth, delta=0.5):
    x = AngularDistanceLoss(pred, truth, reduction="none")
    x = (delta**2) * (torch.sqrt(1 + (x/delta)**2) - 1)
    return x.mean()

def TukeyAngularDistanceLoss(pred, truth, delta=0.01):
    x = AngularDistanceLoss(pred, truth, reduction="none")
    for i in x:
        if i < delta:
            i = (delta**2 / 6) * (1 - (1 - (i/delta)**2)**3)
        else:
            i = (delta**2 / 6)
    return x.mean()
    
def get_p_of_bins(metric, es, bins, p):
    indices = np.digitize(es, bins)
    ps = []
    for i in range(1, bins.shape[0] + 1):
        ps.append(np.percentile(metric[np.where(indices == i)], p))
    return np.array(ps)

def get_mean_of_bins(metric, es, bins):
    indices = np.digitize(es, bins)
    ps = []
    for i in range(1, bins.shape[0] + 1):
        ps.append(np.mean(metric[np.where(indices == i)]))
    return np.array(ps)

def make_double_plot(y1, y2, bins, xlabel, y1label, y2label, savename):
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(bins, y1, 'g-')
    ax2.plot(bins, y2, 'b-')

    ax1.set_xlabel(xlabel)
    ax1.set_xscale('log')
    ax1.set_ylabel(y1label, color='g')
    ax2.set_ylabel(y2label, color='b')
    plt.savefig(savename)

def generate_hybrid_kernel(D, ks, stride, dil):
    dimension = D
    kernel_size = [ks, ks, ks, ks]
    axis_types = [ME.RegionType.HYPER_CUBE, ME.RegionType.HYPER_CUBE, ME.RegionType.HYPER_CUBE, ME.RegionType.HYPER_CROSS]
    center = True
    dilation = [dil, dil, dil, dil]
    tensor_stride = [stride, stride, stride, stride]
    up_stride = [stride, stride, stride, stride]

    import math

    region_offset = [
                [
                    0,
                ]
                * dimension
            ]
    kernel_size_list = kernel_size
    # First HYPER_CUBE
    for axis_type, curr_kernel_size, d in zip(
        axis_types, kernel_size_list, range(dimension)
    ):
        new_offset = []
        if axis_type == ME.RegionType.HYPER_CUBE:
            for offset in region_offset:
                for curr_offset in range(curr_kernel_size):
                    off_center = (
                        int(math.floor((curr_kernel_size - 1) / 2)) if center else 0
                    )
                    offset = offset.copy()  # Do not modify the original
                    # Exclude the coord (0, 0, ..., 0)
                    if curr_offset == off_center:
                        continue
                    offset[d] = (
                        (curr_offset - off_center)
                        * dilation[d]
                        * (tensor_stride[d] / up_stride[d])
                    )
                    new_offset.append(offset)
        region_offset.extend(new_offset)

    # Second, HYPER_CROSS
    for axis_type, curr_kernel_size, d in zip(
        axis_types, kernel_size_list, range(dimension)
    ):
        new_offset = []
        if axis_type == ME.RegionType.HYPER_CROSS:
            for curr_offset in range(curr_kernel_size):
                off_center = (
                    int(math.floor((curr_kernel_size - 1) / 2)) if center else 0
                )
                offset = [
                    0,
                ] * dimension
                # Exclude the coord (0, 0, ..., 0)
                if curr_offset == off_center:
                    continue
                offset[d] = (
                    (curr_offset - off_center)
                    * dilation[d]
                    * (tensor_stride[d] / up_stride[d])
                )
                new_offset.append(offset)
        region_offset.extend(new_offset)
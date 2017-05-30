import torch
import torch.nn as nn

def distance_matrix_vector(anchor, positive):
    """Given batch of anchor descriptors and positive descriptors calculate distance matrix"""

    d1_sq = torch.sum(anchor * anchor, dim=1)
    d2_sq = torch.sum(positive * positive, dim=1)

    eps = 1e-6
    return torch.sqrt((d1_sq.repeat(1, anchor.size(0)) + torch.t(d2_sq.repeat(1, positive.size(0)))
                      - 2.0 * torch.bmm(anchor.unsqueeze(0), torch.t(positive).unsqueeze(0)).squeeze(0))+eps)

def loss_margin_min(anchor, positive, anchor_swap = False, anchor_ave = False, margin = 1.0):
    """HardNet margin loss - calculates loss based on distance matrix based on positive distance and closest negative distance.
    """

    assert anchor.size() == positive.size(), "Input sizes between positive and negative must be equal."
    assert anchor.dim() == 2, "Inputd must be a 2D matrix."

    dist_matrix = distance_matrix_vector(anchor, positive)
    eye = torch.autograd.Variable(torch.eye(dist_matrix.size(1))).cuda()

    # steps to filter out same patches that occur in distance matrix as negatives
    pos = torch.diag(dist_matrix)
    dist_without_min_on_diag = dist_matrix+eye*10
    mask = (dist_without_min_on_diag.ge(0.008)-1)*-1
    mask = mask.type_as(dist_without_min_on_diag)*10
    dist_without_min_on_diag = dist_without_min_on_diag+mask

    min_neg = torch.min(dist_without_min_on_diag,1)[0]
    if anchor_swap:
        min_neg2 = torch.t(torch.min(dist_without_min_on_diag,0)[0])
        min_neg = torch.min(min_neg,min_neg2)
    min_neg = torch.t(min_neg).squeeze(0)
    dist_hinge = torch.clamp(margin + pos - min_neg, min=0.0)

    if anchor_ave:
        min_neg2 = torch.t(torch.min(dist_without_min_on_diag,0)[0])
        min_neg2 = torch.t(min_neg2).squeeze(0)
        dist_hinge2 = torch.clamp(1.0 + pos - min_neg2, min=0.0)
        dist_hinge = 0.5 * (dist_hinge2 + dist_hinge)

    loss = torch.mean(dist_hinge)
    return loss
import torch
import torch.nn as nn
import sys

def distance_matrix_vector(anchor, positive):
    """Given batch of anchor descriptors and positive descriptors calculate distance matrix"""

    d1_sq = torch.sum(anchor * anchor, dim=1)
    d2_sq = torch.sum(positive * positive, dim=1)

    eps = 1e-6
    return torch.sqrt((d1_sq.repeat(1, anchor.size(0)) + torch.t(d2_sq.repeat(1, positive.size(0)))
                      - 2.0 * torch.bmm(anchor.unsqueeze(0), torch.t(positive).unsqueeze(0)).squeeze(0))+eps)

def distance_vectors_pairwise(anchor, positive, negative):
    """Given batch of anchor descriptors and positive descriptors calculate distance matrix"""

    a_sq = torch.sum(anchor * anchor, dim=1)
    p_sq = torch.sum(positive * positive, dim=1)
    n_sq = torch.sum(negative * negative, dim=1)

    eps = 1e-6
    d_a_p = torch.sqrt(a_sq + p_sq - 2*torch.sum(anchor * positive, dim = 1) + eps)
    d_a_n = torch.sqrt(a_sq + n_sq - 2*torch.sum(anchor * negative, dim = 1) + eps)
    d_p_n = torch.sqrt(p_sq + n_sq - 2*torch.sum(positive * negative, dim = 1) + eps)
    return d_a_p, d_a_n, d_p_n
def loss_random_sampling(anchor, positive, negative, anchor_swap = False, margin = 1.0, loss_type = "triplet_margin"):
    """Loss with random sampling (no hard in batch).
    """

    assert anchor.size() == positive.size(), "Input sizes between positive and negative must be equal."
    assert anchor.size() == negative.size(), "Input sizes between positive and negative must be equal."
    assert anchor.dim() == 2, "Inputd must be a 2D matrix."
    eps = 1e-8
    (pos, d_a_n, d_p_n) = distance_vectors_pairwise(anchor, positive, negative)
    if anchor_swap:
       min_neg = torch.min(d_a_n, d_p_n)
    else:
       min_neg = d_a_n

    if loss_type == "triplet_margin":
        loss = torch.clamp(margin + pos - min_neg, min=0.0)
    elif loss_type == 'softmax':
        exp_pos = torch.exp(2.0 - pos);
        exp_den = exp_pos + torch.exp(2.0 - min_neg) + eps;
        loss = - torch.log( exp_pos / exp_den )
    elif loss_type == 'contrastive':
        loss = torch.clamp(margin - min_neg, min=0.0) + pos;
    else: 
        print ('Unknown loss type. Try triplet_margin, softmax or contrastive')
        sys.exit(1)
    loss = torch.mean(loss)
    return loss

def loss_HardNet(anchor, positive, anchor_swap = False, anchor_ave = False, margin = 1.0, batch_reduce = 'min', loss_type = "triplet_margin"):
    """HardNet margin loss - calculates loss based on distance matrix based on positive distance and closest negative distance.
    """

    assert anchor.size() == positive.size(), "Input sizes between positive and negative must be equal."
    assert anchor.dim() == 2, "Inputd must be a 2D matrix."
    eps = 1e-8
    dist_matrix = distance_matrix_vector(anchor, positive)
    eye = torch.autograd.Variable(torch.eye(dist_matrix.size(1))).cuda()

    # steps to filter out same patches that occur in distance matrix as negatives
    pos1 = torch.diag(dist_matrix)
    dist_without_min_on_diag = dist_matrix+eye*10
    mask = (dist_without_min_on_diag.ge(0.008)-1)*-1
    mask = mask.type_as(dist_without_min_on_diag)*10
    dist_without_min_on_diag = dist_without_min_on_diag+mask
    if batch_reduce == 'min':
        min_neg = torch.min(dist_without_min_on_diag,1)[0]
        if anchor_swap:
            min_neg2 = torch.t(torch.min(dist_without_min_on_diag,0)[0])
            min_neg = torch.min(min_neg,min_neg2)
        min_neg = torch.t(min_neg).squeeze(0)
        pos = pos1
    elif batch_reduce == 'average':
        pos = pos1.repeat(anchor.size(0)).view(-1,1).squeeze(0)
        min_neg = dist_without_min_on_diag.view(-1,1)
        if anchor_swap:
            min_neg2 = torch.t(dist_without_min_on_diag).contiguous().view(-1,1)
            min_neg = torch.min(min_neg,min_neg2)
        min_neg = min_neg.squeeze(0)
    elif batch_reduce == 'random':
        idxs = torch.autograd.Variable(torch.randperm(anchor.size()[0]).long()).cuda()
        min_neg = dist_without_min_on_diag.gather(1,idxs.view(-1,1))
        if anchor_swap:
            min_neg2 = torch.t(dist_without_min_on_diag).gather(1,idxs.view(-1,1)) 
            min_neg = torch.min(min_neg,min_neg2)
        min_neg = torch.t(min_neg).squeeze(0)
        pos = pos1
    else: 
        print ('Unknown batch reduce mode. Try min, average or random')
        sys.exit(1)

    if loss_type == "triplet_margin":
        loss = torch.clamp(margin + pos - min_neg, min=0.0)
    elif loss_type == 'softmax':
        exp_pos = torch.exp(2.0 - pos);
        exp_den = exp_pos + torch.exp(2.0 - min_neg) + eps;
        loss = - torch.log( exp_pos / exp_den )
    elif loss_type == 'contrastive':
        loss = torch.clamp(margin - min_neg, min=0.0) + pos;
    else: 
        print ('Unknown loss type. Try triplet_margin, softmax or contrastive')
        sys.exit(1)
    loss = torch.mean(loss)

    return loss


from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertEncoder
from transformers.file_utils import ModelOutput
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def unique(x, dim=None):
    """Unique elements of x and indices of those unique elements
    https://github.com/pytorch/pytorch/issues/36748#issuecomment-619514810

    e.g.

    unique(tensor([
        [1, 2, 3],
        [1, 2, 4],
        [1, 2, 3],
        [1, 2, 5]
    ]), dim=0)
    => (tensor([[1, 2, 3],
                [1, 2, 4],
                [1, 2, 5]]),
        tensor([0, 1, 3]))
    """
    unique, inverse = torch.unique(
        x, sorted=True, return_inverse=True, dim=dim)
    perm = torch.arange(inverse.size(0), dtype=inverse.dtype,
                        device=inverse.device)
    inverse, perm = inverse.flip([0]), perm.flip([0])
    return unique, inverse.new_empty(unique.size(0)).scatter_(0, inverse, perm)


class HDCL(nn.Module):
    def __init__(self, temperature=0.07,
                 base_temperature=0.07, layer_penalty=None, label_depths=None, device='cuda'):
        super(HDCL, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.device = device
        self.label_depths = label_depths  # shape (num_labels, 1)
        if not layer_penalty:
            self.layer_penalty = self.pow_2
        else:
            self.layer_penalty = layer_penalty
        self.sup_con_loss = WeightSupConLoss(temperature, device=device)

    def pow_2(self, value):
        return torch.pow(2, value)

    def forward(self, features, labels):
        """
        :param features: shape (batch_size, num_labels, feature_dim)
        :param labels: shape (batch_size, num_labels)
        """
        device = labels.device
        self.label_depths = self.label_depths.to(device)

        # mask = torch.ones(labels.shape).to(device) # shape (batch_size, 4 <- hierarchy)
        mask = labels.clone().detach().to(device)
        cumulative_loss = torch.tensor(0.0).to(device)
        max_loss_lower_layer = torch.tensor(float('-inf'))
        max_depths = int(torch.max(self.label_depths).item()) + 1
        # capture the loss for each layer by create a list of losses
        loss_by_depths = []
        mask_by_depths = []
        for l in range(1, max_depths):
            # get depth_mask with those smaller than max_depths - l
            depth_mask = self.label_depths <= (max_depths - l)       # shape (num_labels, 1)
            # check the device for depth_mask and mask
            assert depth_mask.device == mask.device, f'depth_mask.device: {depth_mask.device}, mask.device: {mask.device}'
            # filter the mask with depth_mask
            mask = mask * depth_mask # shape (batch_size, num_labels)
            # mask[:, labels.shape[1]-l:] = 0
            layer_labels = labels * mask

            label_overlap = torch.matmul(layer_labels.float(), layer_labels.float().T)
            label_union = torch.sum(layer_labels.float(), dim=1, keepdim=True) + \
                          torch.sum(layer_labels.float(), dim=1, keepdim=True).T - label_overlap
            label_overlap_ratio = label_overlap / (label_union + 1e-8)
            # positive if label_overlap>0
            mask_labels = label_overlap > 0

            weights = label_overlap_ratio / torch.sum(label_overlap_ratio, dim=1, keepdim=True)

            layer_loss = self.sup_con_loss(features, mask=mask_labels, weights=weights)
            mask_by_depths.insert(0, mask_labels.detach().cpu().numpy())

            layer_loss = torch.max(max_loss_lower_layer.to(layer_loss.device), layer_loss)
            tmp = torch.tensor(1 / (l)).type(torch.float).to(layer_loss.device)
            layer_loss = self.layer_penalty(tmp) * layer_loss
            cumulative_loss += layer_loss
            loss_by_depths.insert(0, layer_loss.detach().cpu().numpy())

            _, unique_indices = unique(layer_labels, dim=0)
            labels = labels[unique_indices]
            mask = mask[unique_indices]
            features = features[unique_indices]
        return cumulative_loss / labels.shape[1], np.array(loss_by_depths) / labels.shape[1]

    
class WeightSupConLoss(nn.Module):
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07, device='cuda'):
        super(WeightSupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.device = device

    def forward(self, features, labels=None, mask=None, weights=None):
        device = features.device

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]

        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)  # [batch_size, 1]
            if labels.shape[0] != batch_size:
                raise ValueError(f'Num of labels does not match num of features, {labels.shape[0]} != {batch_size}')
            # compute the 2d mask
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        # features: [bsz, n_views, f_dim]
        # mask.shape = [batch_size, batch_size]

        contrast_count = features.shape[1]

        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)  # [batch_size * n_views, feature_size]

        if self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)

        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        eplison = 1e-8
        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask  # [batch_size * n_views, batch_size * n_views]
        log_prob = logits - torch.log(
            exp_logits.sum(1, keepdim=True) + eplison)  # [batch_size * n_views, batch_size * n_views]

        # positive weights
        if weights is not None:
            weights = weights.repeat(anchor_count, contrast_count)
            mean_log_prob_pos = (weights * mask * log_prob).sum(1) / (weights * mask).sum(1)
        else:
            mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + eplison)

        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss




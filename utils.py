import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom
import torch.nn as nn
import SimpleITK as sitk
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):
    def __init__(self,
                 ignore_index=-1,
                 prompt_num=2):
        super(ContrastiveLoss, self).__init__()

        self.ignore_label = ignore_index
        # Retained for caller compatibility; text rows determine the label count.
        self.prompt_num = prompt_num

    def sample_anchor(self, features, label, ignore, n_view):
        batch_size, feat_dim = features.shape[0], features.shape[-1]

        classes = []
        total_classes = 0
        for i in range(batch_size):
            i_label = label[i]

            i_classes = torch.unique(i_label)
            i_classes = [x for x in i_classes if x not in ignore]

            classes.append(i_classes)
            total_classes += len(i_classes)
        if total_classes == 0:
            return None, None

        anc_features = []
        anc_labels = []

        for i in range(batch_size):
            i_label = label[i]
            i_classes = classes[i]

            for cls_id in i_classes:
                indices = (cls_id == i_label).nonzero()
                if indices.shape[0] <= n_view:
                    anc_features.append(features[i][indices].squeeze(1))
                else:
                    keep = torch.randperm(indices.shape[0])[:n_view]
                    indices = indices[keep]
                    anc_features.append(features[i][indices].squeeze(1))
                anc_labels.append(torch.full((indices.shape[0],), cls_id))

        anc_features = torch.cat(anc_features, dim=0)
        anc_labels = torch.cat(anc_labels).to(features.device)

        return anc_features, anc_labels

    def contrastive_loss(self, anchor, target):
        mask = torch.eq(target.unsqueeze(1), target.unsqueeze(0)).float().to(anchor.device)
        sim = torch.div(torch.matmul(anchor, anchor.T), 0.07)
        logits_max, _ = torch.max(sim, dim=1, keepdim=True)
        logits = sim - logits_max.detach()
        neg_mask = 1 - mask

        logits_mask = torch.ones((mask.shape[0], mask.shape[0]), dtype=bool).to(anchor.device)
        logits_mask.fill_diagonal_(0)
        mask = mask * logits_mask

        neg_logits = torch.exp(logits) * neg_mask
        neg_logits = neg_logits.sum(dim=1, keepdim=True)
        exp_logits = torch.exp(logits)

        log_prob = logits - torch.log(exp_logits + neg_logits)
        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / (mask.sum(dim=1) + 1e-5)
        loss = -mean_log_prob_pos.mean()

        return loss

    def forward(self, features_embedding_all, label, text_embedding_all, ignore, sample_num = 10):
        if len(features_embedding_all) != 3 or len(text_embedding_all) != 3:
            raise ValueError("Contrastive loss requires three feature/text scales")
        if label.ndim != 3 or not torch.isfinite(label).all() or not ((label >= 0) & (label <= 3) & (label == label.long())).all():
            raise ValueError("Labels must have shape (B,H,W) and integer class IDs 0..3 (legacy MyoPS380)")
        if sample_num <= 0:
            raise ValueError("sample_num must be positive")
        loss = 0.0
        n_view_list = [5, 2, 1]
        n_view_list = [int(x * sample_num) for x in n_view_list]

        for features, text_embedding, n_view in zip(features_embedding_all, text_embedding_all, n_view_list):
            if features.ndim != 4 or features.shape[0] != label.shape[0]:
                raise ValueError("Feature batches must match the target batch")
            if text_embedding.ndim != 2 or text_embedding.shape[1] != features.shape[1]:
                raise ValueError("Text and image embedding dimensions must match")
            if text_embedding.shape[0] == 0 or text_embedding.shape[0] % 4:
                raise ValueError("Text embeddings must contain complete ordered groups of classes [0,1,2,3]")
            if features.device != label.device or text_embedding.device != features.device:
                raise ValueError("Features, labels and text embeddings must use the same device")
            if n_view < 1:
                raise ValueError("sample_num must produce at least one anchor per scale")
            i_label = label.unsqueeze(1).float()
            i_label = F.interpolate(i_label, (features.shape[2], features.shape[3]), mode='nearest')
            i_label = i_label.long()
            assert i_label.shape[-1] == features.shape[-1], '{} {}'.format(i_label.shape, features.shape)

            batch_size = features.shape[0]
            i_label = i_label.view(batch_size, -1)
            features = features.permute(0, 2, 3, 1).reshape(batch_size, -1, features.shape[1])

            anchor, target = self.sample_anchor(features, i_label, ignore, n_view)
            if anchor is None:
                # This scale has no eligible image anchors. Keep both branches in
                # the graph without changing the reduction of the other scales.
                loss += features.sum() * 0.0 + text_embedding.sum() * 0.0
                continue
            anchor = torch.cat([anchor, text_embedding], dim=0)
            anchor = F.normalize(anchor, p=2, dim=1)
            text_embedding_label = torch.arange(4, device=target.device).repeat(text_embedding.shape[0] // 4)
            target = torch.cat([target, text_embedding_label], dim=0)
            loss += self.contrastive_loss(anchor, target)

        return loss

class SoftmaxWeightedLoss(nn.Module):
    def __init__(self, num_cls=4):
        super(SoftmaxWeightedLoss, self).__init__()
        self.num_cls = num_cls

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.num_cls):
            temp_prob = input_tensor == i
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def forward(self, output, target):
        target = target.float()
        B, _, H, W = output.size()
        target = self._one_hot_encoder(target)
        for i in range(self.num_cls):
            outputi = output[:, i, :, :]
            targeti = target[:, i, :, :]
            weighted = 1.0 - (torch.sum(targeti, (1, 2)) * 1.0 / torch.sum(target, (1, 2, 3)))
            weighted = torch.reshape(weighted, (-1, 1, 1)).repeat(1, H, W)
            if i == 0:
                cross_loss = 0
            else:
                cross_loss += -1.0 * weighted * targeti * torch.log(torch.clamp(outputi, min=0.005, max=1)).float()
        cross_loss = torch.mean(cross_loss)
        return cross_loss

class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()
    
    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        dice = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - dice
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
            weight[0] = 0.5
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            loss_label = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - loss_label.item())
            loss += loss_label * weight[i]
        return loss / self.n_classes


def calculate_metric_percase(pred, gt):
    """Read foreground masks without modifying them; empty metrics are NaN."""
    pred, gt = np.asarray(pred), np.asarray(gt)
    if pred.shape != gt.shape:
        raise ValueError("Prediction and GT masks must have the same shape")
    if not np.isfinite(pred).all() or not np.isfinite(gt).all():
        raise ValueError("Prediction and GT masks must not contain NaN or Inf")
    pred, gt = pred.astype(bool, copy=False), gt.astype(bool, copy=False)
    pred_empty = not np.any(pred)
    gt_empty = not np.any(gt)
    if pred_empty and gt_empty:
        return np.nan, np.nan
    if pred_empty or gt_empty:
        return 0.0, np.nan
    return metric.binary.dc(pred, gt), metric.binary.hd95(pred, gt)


def evaluation_regions(classes):
    """Keep the original order of individual classes and merged regions."""
    regions = []
    for i in range(1, classes):
        regions.append((i,))
        if i == 3:
            regions.extend([(2, 3), tuple(range(1, classes))])
    return regions


def calculate_volume_metrics(prediction, label, classes):
    """Build independent binary region masks, preserving both label maps."""
    prediction, label = np.asarray(prediction), np.asarray(label)
    if prediction.shape != label.shape:
        raise ValueError("Prediction and GT label maps must have the same shape")
    if not np.isin(prediction, range(classes)).all() or not np.isin(label, range(classes)).all():
        raise ValueError(f"Prediction and GT must contain only class IDs 0..{classes - 1}")
    return [calculate_metric_percase(np.isin(prediction, ids), np.isin(label, ids))
            for ids in evaluation_regions(classes)]


def aggregate_case_metrics(case_metrics):
    """Return per-region Dice/HD95 means and their separate non-NaN counts."""
    values = np.asarray(case_metrics, dtype=float)
    if values.ndim != 3 or values.shape[0] == 0 or values.shape[2] != 2:
        raise ValueError("Expected at least one case with shape (cases, regions, 2)")
    if np.isinf(values).any():
        raise ValueError("Case metrics must not contain Inf")
    valid_counts = np.sum(~np.isnan(values), axis=0)
    means = np.full(values.shape[1:], np.nan, dtype=float)
    np.divide(np.nansum(values, axis=0), valid_counts,
              out=means, where=valid_counts > 0)
    return means, valid_counts


def _prediction_from_logits(outputs, case, slice_index=None):
    if not torch.isfinite(outputs).all():
        location = f"case {case!r}"
        if slice_index is not None:
            location += f", slice {slice_index}"
        raise ValueError(f"Model output contains NaN or Inf for {location}")
    out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
    return out.cpu().detach().numpy()


def test_single_volume(image, image1, image2, label, net, classes, patch_size=[256, 256], test_save_path=None, case=None, z_spacing=1, return_prediction=False):
    """Evaluate without changing input tensors; optionally return the label map."""
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
    image1, image2 = image1.squeeze(0).cpu().detach().numpy(), image2.squeeze(0).cpu().detach().numpy()
    if len(image.shape) == 3:  
        prediction = np.zeros_like(label)

        for ind in range(image.shape[2]):
            slice = image[:, :, ind]
            slice1 = image1[:, :, ind]
            slice2 = image2[:, :, ind]
            x, y = slice.shape[0], slice.shape[1]

            if x != patch_size[0] or y != patch_size[1]:
                slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=3)  
                slice1 = zoom(slice1, (patch_size[0] / x, patch_size[1] / y), order=3)
                slice2 = zoom(slice2, (patch_size[0] / x, patch_size[1] / y), order=3)
            input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()  
            input1 = torch.from_numpy(slice1).unsqueeze(0).unsqueeze(0).float().cuda()
            input2 = torch.from_numpy(slice2).unsqueeze(0).unsqueeze(0).float().cuda()
            net.eval()
            with torch.no_grad():
                outputs = net(input,input1,input2)
                out = _prediction_from_logits(outputs, case, ind)

                if x != patch_size[0] or y != patch_size[1]:
                    pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0) 
                else:
                    pred = out
                prediction[:, :, ind] = pred  
    else: 
        input = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float().cuda()
        input1 = torch.from_numpy(image1).unsqueeze(0).unsqueeze(0).float().cuda()
        input2 = torch.from_numpy(image2).unsqueeze(0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            prediction = _prediction_from_logits(net(input,input1,input2), case)

    metric_list = calculate_volume_metrics(prediction, label, classes)
    
    if test_save_path is not None:
            img_itk = sitk.GetImageFromArray(image.transpose(2,0,1).astype(np.float32))
            prd_itk = sitk.GetImageFromArray(prediction.transpose(2,0,1).astype(np.float32))
            lab_itk = sitk.GetImageFromArray(label.transpose(2,0,1).astype(np.float32))
            img_itk.SetSpacing((z_spacing, 1, 1))
            prd_itk.SetSpacing((z_spacing, 1, 1))
            lab_itk.SetSpacing((z_spacing, 1, 1))
            sitk.WriteImage(prd_itk, test_save_path + '/'+case + "_pred.nii.gz")
            sitk.WriteImage(img_itk, test_save_path + '/'+ case + "_img.nii.gz")
            sitk.WriteImage(lab_itk, test_save_path + '/'+ case + "_gt.nii.gz")

    if return_prediction:
        return metric_list, prediction
    return metric_list

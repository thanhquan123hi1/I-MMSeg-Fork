"""Unmodified training functions from the baseline commit, for regression checks only."""
import logging, os, random, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from torchvision import transforms
from tqdm import tqdm

class ContrastiveLoss(nn.Module):
    def __init__(self,
                 ignore_index=-1,
                 prompt_num=2):
        super(ContrastiveLoss, self).__init__()

        self.ignore_label = ignore_index
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
        loss = 0.0
        n_view_list = [5, 2, 1]
        n_view_list = [int(x * sample_num) for x in n_view_list]

        for features, text_embedding, n_view in zip(features_embedding_all, text_embedding_all, n_view_list):
            i_label = label.unsqueeze(1).float()
            i_label = F.interpolate(i_label, (features.shape[2], features.shape[3]), mode='nearest')
            i_label = i_label.long()
            assert i_label.shape[-1] == features.shape[-1], '{} {}'.format(i_label.shape, features.shape)

            batch_size = features.shape[0]
            i_label = i_label.view(batch_size, -1)
            features = features.permute(0, 2, 3, 1).reshape(batch_size, -1, features.shape[1])

            anchor, target = self.sample_anchor(features, i_label, ignore, n_view)
            anchor = torch.cat([anchor, text_embedding], dim=0)
            anchor = F.normalize(anchor, p=2, dim=1)
            text_embedding_label = torch.tensor([0, 1, 2, 3]).repeat(self.prompt_num).to(target.device)
            target = torch.cat([target, text_embedding_label], dim=0)
            loss += self.contrastive_loss(anchor, target)

        return loss

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
        print(class_wise_dice)
        return loss / self.n_classes

def trainer_Myops(args, model, snapshot_path):
    from datasets.dataset_Myops import Myops_dataset, RandomGenerator
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    db_train = Myops_dataset(base_dir=args.root_path, base_dir1=args.root_path1, base_dir2=args.root_path2, list_dir=args.list_dir, split="train",
                               transform=transforms.Compose(
                                   [RandomGenerator(output_size=[args.img_size, args.img_size])]))
    print("The length of train set is: {}".format(len(db_train)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()
    con_loss = ContrastiveLoss()
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)
    optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=0.0001)
    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader) 
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        do_contrast = epoch_num > args.start_contrast_epoch
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, image1_batch, image2_batch, label_batch = sampled_batch['image'], sampled_batch['image1'], sampled_batch['image2'], sampled_batch['label']
            image_batch, image1_batch, image2_batch, label_batch = image_batch.cuda(), image1_batch.cuda(), image2_batch.cuda(), label_batch.cuda()
            out_pre, dec_seg, features_embedding_list, text_embedding_list= model(image_batch, image1_batch, image2_batch, do_contrast)
            ignores = ([2,3],[0],[0])
            loss_all = 0
            if do_contrast:
                for i in range(len(features_embedding_list)):
                    feature_list = features_embedding_list[i]
                    ignore = ignores[i]
                    loss_con = con_loss(feature_list,
                                        label_batch,
                                        text_embedding_list,
                                        ignore,
                                        sample_num = args.contrast_sample_num,
                                        )
                    loss_all += loss_con
                loss_all = loss_all /len(features_embedding_list)
            else:
                loss_all = 0
            out_cross_loss = ce_loss(out_pre, label_batch)
            out_dice_loss = dice_loss(out_pre, label_batch, softmax=True)
            out_loss = 0.2* out_cross_loss + 0.8* out_dice_loss
            dec_cross_loss = torch.zeros(1).cuda().float()
            dec_dice_loss = torch.zeros(1).cuda().float()
            for dec_pred in dec_seg:
                dec_cross_loss += ce_loss(dec_pred, label_batch)
                dec_dice_loss += dice_loss(dec_pred, label_batch, softmax=True)
            dec_loss = 0.2* dec_cross_loss + 0.8* dec_dice_loss
            
            if epoch_num < args.region_fusion_start_epoch:
                loss = out_loss * 0.0 + dec_loss+ loss_all * args.contrast_w
            else:
                loss = out_loss + 0.5 * dec_loss+ 0.5 * loss_all * args.contrast_w

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_
            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', out_dice_loss, iter_num)

            logging.info('iteration %d : loss : %f, loss_fuse: %f' % (iter_num, loss.item(), out_dice_loss.item()))

            if iter_num % 20 == 0:
                image = image_batch[1, 0:1, :, :]
                image = (image - image.min()) / (image.max() - image.min())
                writer.add_image('train/Image', image, iter_num)
                out_pre = torch.argmax(torch.softmax(out_pre, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction', out_pre[1, ...] * 50, iter_num)
                labs = label_batch[1, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)

        save_interval = 20  
        if (epoch_num + 1) % save_interval == 0:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

        if epoch_num >= max_epoch - 5:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            iterator.close()
        if epoch_num >=max_epoch:    
            break



    writer.close()
    return "Training Finished!"

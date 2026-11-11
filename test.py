import argparse
import logging
import os
import random
import sys
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import aggregate_case_metrics, evaluation_regions, test_single_volume
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from collections import OrderedDict
from pathlib import Path
from datasets.profiles import build_dataset, dataset_name, default_root

PROJECT_ROOT = Path(__file__).resolve().parent

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=dataset_name, default='Myops', help='MyoPS380 (Myops) or MyoPS++')
parser.add_argument('--data-root', '--data_root', type=Path, help='Processed MyoPS380, MyoPS++ ROI export, or SCAR ROI cache')
parser.add_argument('--num_classes', type=int,
                    default=4, help='output channel of network')
parser.add_argument('--list_dir', type=str,
                    default=None, help='MyoPS380 split manifest directory')

parser.add_argument('--max_iterations', type=int,default=30000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int, default=300, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=24,
                    help='batch_size per gpu')

parser.add_argument('--img_size', type=int, default=128, help='input patch size of network input')
parser.add_argument('--is_savenii', action="store_true", default="True", help='whether to save results during inference')
parser.add_argument('--n_skip', type=int, default=3, help='using number of skip-connect, default is num')
parser.add_argument('--vit_name', type=str, default='R50-ViT-B_16', help='select one vit model')

parser.add_argument('--test_save_dir', type=str, default='./predictions', help='saving prediction as nii!')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.001, help='segmentation network learning rate')
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--vit_patches_size', type=int, default=16, help='vit_patches_size, default is 16')
parser.add_argument('--checkpoint', type=Path, help='Explicit I-MMSeg checkpoint')


def inference(args, model, test_save_path=None):
    db_test = build_dataset(args.dataset, args.data_root, "test_vol", args.list_dir)
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=0)
    unit = "mm" if args.dataset == "MyoPS++" else "voxel"
    logging.info("Regions (model order): normal, scar, edema-only, edema including scar, whole myocardium; HD95 unit=%s", unit)
    logging.info("{} test iterations per epoch".format(len(testloader)))
    model.eval()
    case_metrics = []
    regions = evaluation_regions(args.num_classes)
    empty_counts = np.zeros((len(regions), 3), dtype=int)
    for i_batch, sampled_batch in tqdm(enumerate(testloader)):
        image, iamge1, image2, label, case_name = sampled_batch["image"], sampled_batch["image1"], sampled_batch["image2"], sampled_batch["label"], sampled_batch['case_name'][0]
        metric_i, prediction = test_single_volume(image, iamge1, image2, label, model, classes=args.num_classes, patch_size=[args.img_size, args.img_size],
                                      test_save_path=test_save_path, case=case_name, return_prediction=True,
                                      spacing=sampled_batch['spacing'][0].numpy() if 'spacing' in sampled_batch else None,
                                      roi_affine=sampled_batch['affine'][0].numpy() if 'affine' in sampled_batch else None)
        gt = label.squeeze(0).cpu().detach().numpy()
        for idx, (dice, hd95) in enumerate(metric_i):
            logging.info(f"Class {idx + 1} - Dice: {dice:.4f}, HD95: {hd95:.4f}")
            pred_empty = not np.isin(prediction, regions[idx]).any()
            gt_empty = not np.isin(gt, regions[idx]).any()
            empty_counts[idx] += (pred_empty and gt_empty,
                                  pred_empty and not gt_empty,
                                  gt_empty and not pred_empty)
        case_metrics.append(metric_i)
    metric_list, valid_counts = aggregate_case_metrics(case_metrics)
    for i in range(1, args.num_classes+2):
        logging.info('Mean class %d mean_dice %f mean_hd95 %f' % (i, metric_list[i-1][0], metric_list[i-1][1]))
        logging.info('Mean class %d valid_dice_cases %d valid_hd95_cases %d total_cases %d' %
                     (i, valid_counts[i-1][0], valid_counts[i-1][1], len(db_test)))
        logging.info('Mean class %d both_empty_cases %d missed_foreground_cases %d false_positive_foreground_cases %d' %
                     (i, empty_counts[i-1][0], empty_counts[i-1][1], empty_counts[i-1][2]))
    return "Testing Finished!"

if __name__ == "__main__":
    args = parser.parse_args()
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    args.data_root = args.data_root or default_root(args.dataset)
    args.num_classes = 4
    if args.dataset == "MyoPS++" and args.img_size != 128:
        parser.error("The MyoPS++ ROI128 profile requires --img_size 128")
    args.is_pretrain = True

    # name the same snapshot defined in train script!
    args.exp = 'TU_' + args.dataset + str(args.img_size)
    snapshot_path = f"{PROJECT_ROOT}/weights/{args.exp}/IMMSeg_{args.seed}"

    config_vit = CONFIGS_ViT_seg[args.vit_name]
    config_vit.n_classes = args.num_classes
    config_vit.n_skip = args.n_skip
    config_vit.patches.size = (args.vit_patches_size, args.vit_patches_size)
    if args.vit_name.find('R50') !=-1:
        config_vit.patches.grid = (int(args.img_size/args.vit_patches_size), int(args.img_size/args.vit_patches_size))
    net = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).cuda()

    snapshot = args.checkpoint or f"{snapshot_path}/epoch_{args.max_epochs - 1}.pth"
    checkpoint = torch.load(snapshot, map_location="cpu", weights_only=True)
    new_state_dict = OrderedDict()
    for k, v in checkpoint.items():
        name = k[7:] if k.startswith("module.") else k
        new_state_dict[name] = v
    net.load_state_dict(new_state_dict)
    snapshot_name = snapshot_path.split('/')[-1]

    log_folder = './test_log/' + args.exp
    os.makedirs(log_folder, exist_ok=True)
    logging.basicConfig(filename=log_folder + '/'+snapshot_name+".txt", level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    logging.info(snapshot_name)

    if args.is_savenii:
        test_save_path = os.path.join(args.test_save_dir, args.exp, snapshot_name)
        os.makedirs(test_save_path, exist_ok=True)
    else:
        test_save_path = None
    inference(args, net, test_save_path)

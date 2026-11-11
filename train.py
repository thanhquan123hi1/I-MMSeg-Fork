import argparse
import logging
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from trainer import trainer_Myops
from pathlib import Path
from datasets.profiles import dataset_name, default_root

PROJECT_ROOT = Path(__file__).resolve().parent

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=dataset_name, default='Myops', help='MyoPS380 (Myops) or MyoPS++')
parser.add_argument('--data-root', '--data_root', type=Path, help='Processed MyoPS380, MyoPS++ ROI export, or SCAR ROI cache')
parser.add_argument('--list_dir', type=str,
                    default=None, help='MyoPS380 split manifest directory')
parser.add_argument('--num_classes', type=int,
                    default=4, help='output channel of network')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int,
                    default=300, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,
                    default=16, help='batch_size per gpu (total batch is batch_size * n_gpu)')
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.0003333,
                    help='segmentation network learning rate')
parser.add_argument('--img_size', type=int,
                    default=128, help='input patch size of network input')
parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')
parser.add_argument('--n_skip', type=int,
                    default=3, help='using number of skip-connect, default is num')
parser.add_argument('--vit_name', type=str,
                    default='R50-ViT-B_16', help='select one vit model')
parser.add_argument('--vit_patches_size', type=int,
                    default=16, help='vit_patches_size, default is 16')
parser.add_argument('--region_fusion_start_epoch', default=100, type=int)
parser.add_argument('--start_contrast_epoch', type=int, default=20, help='start contrastive loss epoch')
parser.add_argument('--contrast_sample_num', type=int, default=10, help='contrastive sample_num')
parser.add_argument('--contrast_w', type=float, default=0.1, help='contrastive loss weight')

args = parser.parse_args()


MRI_prompts = [
    'In the bSSFP sequence of cardiac MRI, no disease signals are detected in the myocardial , myocardium is more dark relative to the heart chamber blood signal',
    'In the LGE sequence of cardiac MRI, scar signals enhanced brightening on the myocardial , myocardium is more dark relative to the heart chamber blood signal',
    'In the T2w sequence of cardiac MRI, edema signals enhanced brightening on the myocardial , myocardium is more bright relative to the heart chamber blood signal',
]

class_prompts = [
    'this is a background class',
    'this is a myocardial class',
    'this is a scar class',
    'this is a edema class',
]

device = "cuda" if torch.cuda.is_available() else "cpu"

def Biomedclip_model():
    import open_clip
    from open_clip import get_tokenizer
    biomedclip_model = open_clip.create_model_from_pretrained(
        'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
    biomedclip_model = biomedclip_model[0].text
    biomedclip_model = biomedclip_model.cuda()
    biomedclip_model.eval()
    tokenizer = get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')

    tokenized_prompt1 = MRI_prompts
    for i, prompt in enumerate(tokenized_prompt1):
        tokenized_prompt1[i] = tokenizer(prompt)
    text_input1 = torch.cat(tokenized_prompt1, dim=0).to(device)
    with torch.no_grad():
        text_feature1 = biomedclip_model(text_input1).float()
        torch.save(text_feature1, f'{PROJECT_ROOT}/text_features/embedding_MRI_information.pth')
    
    tokenized_prompt2 = class_prompts
    for i, prompt in enumerate(tokenized_prompt2):
        tokenized_prompt2[i] = tokenizer(prompt)
    text_input2 = torch.cat(tokenized_prompt2, dim=0).to(device)
    with torch.no_grad():
        text_feature2 = biomedclip_model(text_input2).float()
        torch.save(text_feature2, f'{PROJECT_ROOT}/text_features/embedding_class_information.pth')

if __name__ == "__main__":
    # Biomedclip_model()
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
    args.exp = 'TU_' + args.dataset + str(args.img_size)
    snapshot_path = f"{PROJECT_ROOT}/weights/{args.exp}/IMMSeg_{args.seed}"

    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    config_vit = CONFIGS_ViT_seg[args.vit_name]
    config_vit.n_classes = args.num_classes
    config_vit.n_skip = args.n_skip
    if args.vit_name.find('R50') != -1:
        config_vit.patches.grid = (int(args.img_size / args.vit_patches_size), int(args.img_size / args.vit_patches_size))
    net = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).cuda()
    trainer_Myops(args, net, snapshot_path)

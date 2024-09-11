"""make variations of input image"""

import argparse, os, sys, glob
import PIL
import torch
import numpy as np
import torchvision
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange, repeat
from torchvision.utils import make_grid
from torch import autocast
from contextlib import nullcontext
import time
from pytorch_lightning import seed_everything
from ldm.models.diffusion.ddpm import LatentDiffusionSR
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from diffusers import UNet2DConditionModel
from basicsr.metrics import calculate_niqe
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import math
import copy
import shutil
import cv2
from thop import profile

def calc_mean_std(feat, eps=1e-5):
    """Calculate mean and std for adaptive_instance_normalization.
    Args:
        feat (Tensor): 4D tensor.
        eps (float): A small value added to the variance to avoid
            divide-by-zero. Default: 1e-5.
    """
    size = feat.size()
    assert len(size) == 4, 'The input feature should be 4D tensor.'
    b, c = size[:2]
    feat_var = feat.view(b, c, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(b, c, 1, 1)
    feat_mean = feat.view(b, c, -1).mean(dim=2).view(b, c, 1, 1)
    return feat_mean, feat_std

def adaptive_instance_normalization(content_feat, style_feat):
    """Adaptive instance normalization.
    Adjust the reference features to have the similar color and illuminations
    as those in the degradate features.
    Args:
        content_feat (Tensor): The reference feature.
        style_feat (Tensor): The degradate features.
    """
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)
    normalized_feat = (content_feat - content_mean.expand(size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)

def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())

def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model

def load_img(path):
    image = Image.open(path).convert("RGB")
    image=image.resize((512,512))
    w, h = image.size
    print(f"loaded input image of size ({w}, {h}) from {path}")
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    image = image.resize((w, h), resample=PIL.Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.*image - 1.

def compute_psnr(im1, im2):
    p = psnr(im1, im2)
    return p


def compute_ssim(im1, im2):
    isRGB = len(im1.shape) == 3 and im1.shape[-1] == 3
    s = ssim(im1, im2, K1=0.01, K2=0.03, gaussian_weights=True, sigma=1.5, use_sample_covariance=False,
             multichannel=isRGB)
    return s


def count_layers(model):
    layer_count = 0
    for _ in model.modules():
        layer_count += 1
    return layer_count


def main():
    # region
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--init-img",
        type=str,
        nargs="?",
        help="path to the input image",
        default=""
    )
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        default="",
        help="dir to write results to",
    )

    parser.add_argument(
        "--skip_grid",
        action='store_true',
        help="do not save a grid, only individual samples. Helpful when evaluating lots of samples",
    )

    parser.add_argument(
        "--skip_save",
        action='store_true',
        help="do not save indiviual samples. For speed measurements.",
    )

    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=200,
        help="number of ddim sampling steps",
    )

    parser.add_argument(
        "--plms",
        action='store_true',
        help="use plms sampling",
    )
    parser.add_argument(
        "--fixed_code",
        action='store_true',
        help="if enabled, uses the same starting code across all samples ",
    )

    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=1.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=1,
        help="sample this often",
    )
    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor, most often 8 or 16",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=8,
        help="how many samples to produce for each given prompt. A.k.a batch size",
    )
    parser.add_argument(
        "--n_rows",
        type=int,
        default=0,
        help="rows in the grid (default: n_samples)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=5.0,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )

    parser.add_argument(
        "--strength",
        type=float,
        default=1.0,
        help="strength for noising/unnoising. 1.0 corresponds to full destruction of information in init image",
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/bsr_sr/config_sr_finetune.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="", #/raw7/intern/biaowang5/code/LDMSR/logs/2024-04-16T10-12-50_config_sr_finetune/checkpointsepoch_20.pth
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--seed",
        type=int,   
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
    )
    parser.add_argument(
        "--save_input",
        # action='store_true',
        type=bool,
        default=True,
        help="if enabled, save inputs",
    )
    parser.add_argument(
        "--input_size",
        type=int,
        default=512,
        help="input size",
    )
    parser.add_argument(
        "--color_fix",
        action='store_true',
        help="if enabled, use adain for color fix",
    )

    # endregion
    
    opt = parser.parse_args()
    seed_everything(opt.seed)

    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(opt.input_size // 4),
    ])

    config = OmegaConf.load(f"{opt.config}")
    
    torch.cuda.empty_cache()
    
    ########### UNet-model ##############
    # model=LatentDiffusionSR(first_stage_config=config['first_stage_config'],
    #                             cond_stage_config=config['cond_stage_config'],unet_config=config['unet_config'],
    #                             **config['params'])
    
    ########### DiT-model ##############
    # model=LatentDiffusionSR(first_stage_config=config['first_stage_config'],
    #                             cond_stage_config=config['cond_stage_config'],DiT_config=config['DiT_config'],
    #                             **config['params'])
    
    ########### ESRT-model ##############
    model=LatentDiffusionSR(first_stage_config=config['first_stage_config'],
                                cond_stage_config=config['cond_stage_config'],ESRT_config=config['ESRT_config'],
                                **config['params'])
    
    if opt.ckpt:
        checkpoint=torch.load(opt.ckpt,map_location='cpu')
        # model_state_dict=model.state_dict()
        # for k,v in checkpoint["state_dict"].items():
        #     if k in model_state_dict:
        #         model_state_dict[k]=v
        model.load_state_dict(checkpoint['state_dict'])
        print("Model load")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # model = model.to(device)
    
    if os.path.exists(opt.outdir):
        shutil.rmtree(opt.outdir)
        
    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir
    
    batch_size = opt.n_samples
    n_rows = opt.n_rows if opt.n_rows > 0 else batch_size

    sample_path = os.path.join(outpath, "samples")
    os.makedirs(sample_path, exist_ok=True)
    input_path = os.path.join(outpath, "inputs")
    os.makedirs(input_path, exist_ok=True)
    base_count = len(os.listdir(sample_path))
    base_i = len(os.listdir(input_path))
    grid_count = len(os.listdir(outpath)) - 1

    img_list_old = os.listdir(opt.init_img)
    img_list_renew = []
    save_list = os.listdir(sample_path)
    for item in img_list_old:
        if item in save_list:
            pass
        else:
            img_list_renew.append(item)

    img_list = sorted(img_list_renew)
    niters = math.ceil(len(img_list) / batch_size)
    # using list comprehension
    img_list_chunk = [img_list[i * batch_size:(i + 1) * batch_size] for i in range((len(img_list) + batch_size - 1) // batch_size )]
    
    model = model.to(device)
    
    
    layers=count_layers(model)
    print("model layers:",layers)
    
    
    model.eval()
    
    

    # input_size=torch.zeros((1,3,384,384)).to(device)

    # flops, params = profile(model, inputs=(input_size,))

    # print('FLOPs = ' + str(flops/1000**3) + 'G')
    # print('Params = ' + str(params/1000**2) + 'M')

    
    
    if opt.plms:
        raise NotImplementedError("PLMS sampler not (yet) supported")
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model)

    sampler.make_schedule(ddim_num_steps=opt.ddim_steps, ddim_eta=opt.ddim_eta, verbose=False)

    assert 0. <= opt.strength <= 1., 'can only work with strength in [0.0, 1.0]'
    t_enc = int(opt.strength * opt.ddim_steps)
    print(f"target t_enc is {t_enc} steps")

    precision_scope = autocast if opt.precision == "autocast" else nullcontext
    niqe_list = []
    psnr_list=[]
    ssim_list=[]
    x_T = None
    
    with torch.no_grad():
        start_time = time.time()
        with precision_scope("cuda"):
            # with model.ema_scope():
            tic = time.time()
            all_samples = list()
            for n in trange(niters, desc="Sampling"):
                # seed_everything(opt.seed*n)
                cur_img_list = img_list_chunk[n]
                init_image_list = []
                for item in cur_img_list:
                    cur_image = load_img(os.path.join(opt.init_img, item)).to(device)
                    cur_image = transform(cur_image)
                    init_image_list.append(cur_image)
                init_image = torch.cat(init_image_list, dim=0)

                # decode it
                samples, _ = sampler.sample(t_enc, init_image.size(0), (3,opt.input_size // 4,opt.input_size // 4), init_image, eta=opt.ddim_eta, verbose=False, x_T=x_T,
                                            unconditional_guidance_scale=7.5,
                                            unconditional_conditioning=init_image)

                x_samples = model.decode_first_stage(samples)
                if opt.color_fix:
                    x_samples = adaptive_instance_normalization(x_samples, init_image)
                x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
                targets = torch.clamp((init_image+1.0)/2.0, min=0.0, max=1.0)
                if not opt.skip_save:
                    for i in range(x_samples.size(0)):
                        x_sample = 255. * rearrange(x_samples[i].cpu().numpy(), 'c h w -> h w c')
                        niqe_list.append(calculate_niqe(x_sample, 0, input_order='HWC', convert_to='y'))
                        target=255. * rearrange(targets[i].cpu().numpy(), 'c h w -> h w c')
                        target =cv2.resize(target, (512,512), interpolation=cv2.INTER_LINEAR)
                        
                        psnr_list.append(compute_psnr(x_sample[:,:,0].astype(np.uint8),target[:,:,0].astype(np.uint8)))
                        ssim_list.append(compute_ssim(x_sample[:,:,0].astype(np.uint8),target[:,:,0].astype(np.uint8)))
                        
                        Image.fromarray(x_sample.astype(np.uint8)).save(
                            os.path.join(sample_path, cur_img_list[i]))
                    if opt.save_input:
                        for i in range(init_image.size(0)):
                            x_input = 255. * rearrange(init_image[i].cpu().numpy(), 'c h w -> h w c')
                            x_input = (x_input+255.)/2
                            Image.fromarray(x_input.astype(np.uint8)).save(
                                os.path.join(input_path, cur_img_list[i]))
                    base_i += init_image.size(0)
                all_samples.append(x_samples)

            if not opt.skip_grid:
                # additionally, save as grid
                all_samples_new = []
                for item in all_samples:
                    if item.size(0) < batch_size:
                        template_tensor = item[0].unsqueeze(0)
                        add_tensor = torch.zeros_like(template_tensor).repeat(batch_size-item.size(0), 1,1,1)
                        item = torch.cat([item, add_tensor], dim=0)
                        assert item.size(0) == batch_size
                    all_samples_new.append(item)
                grid = torch.stack(all_samples_new, 0)
                grid = rearrange(grid, 'n b c h w -> (n b) c h w')
                grid = make_grid(grid, nrow=n_rows)

                # to image
                grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
                Image.fromarray(grid.astype(np.uint8)).save(os.path.join(outpath, f'grid-{grid_count:04}.png'))
                grid_count += 1

            assert len(niqe_list) == len(img_list)==len(psnr_list)==len(ssim_list)
            avg_niqe = np.mean(np.array(niqe_list))
            avg_npsr=np.mean(np.array(psnr_list))
            avg_ssim=np.mean(np.array(ssim_list))
            print(f"Average NIQE score: {avg_niqe:.3f} \n")
            print(f"Average PNSR score: {avg_npsr:.3f} \n")
            print(f"Average SSIM score: {avg_ssim:.3f} \n")
        
        current_memory = torch.cuda.memory_allocated() / 1e6
        print(f"Current GPU memory usage: {current_memory} MB")
        
        end_time = time.time()

        print(f"Execution time: {(end_time - start_time)/14} seconds")
        
        
    print(f"Your samples are ready and waiting for you here: \n{outpath} \n"
          f" \nEnjoy.")
    
    
    



if __name__ == "__main__":
    main()


import argparse, os, sys, datetime, glob, importlib, csv
import numpy as np
import time
import torch
import torchvision
import pytorch_lightning as pl
import cv2
from pytorch_lightning import seed_everything

from packaging import version
from omegaconf import OmegaConf
from torch.utils.data import random_split, DataLoader, Dataset, Subset,DistributedSampler
from functools import partial
from PIL import Image
import PIL
from torchvision.utils import make_grid
from basicsr.data.realesrgan_dataset import RealESRGANDataset
from ldm.data.base import Txt2ImgIterableBaseDataset
from ldm.models.diffusion.ddpm import LatentDiffusionSR
from ldm.util import instantiate_from_config, instantiate_from_config_sr
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
import matplotlib.pyplot as plt
import logging
from tqdm import tqdm,trange
from einops import rearrange, repeat
from basicsr.metrics import calculate_niqe
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import math
import skimage.color as sc
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler


dist.init_process_group(backend='nccl',)
rank = dist.get_rank()

class WrappedDataset(Dataset):
    """Wraps an arbitrary object with __len__ and __getitem__ into a pytorch dataset"""

    def __init__(self, dataset):
        self.data = dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def worker_init_fn(_):
    worker_info = torch.utils.data.get_worker_info()

    dataset = worker_info.dataset
    worker_id = worker_info.id

    if isinstance(dataset, Txt2ImgIterableBaseDataset):
        split_size = dataset.num_records // worker_info.num_workers
        # reset num_records to the true number to retain reliable length information
        dataset.sample_ids = dataset.valid_ids[worker_id * split_size:(worker_id + 1) * split_size]
        current_id = np.random.choice(len(np.random.get_state()[1]), 1)
        return np.random.seed(np.random.get_state()[1][current_id] + worker_id)
    else:
        return np.random.seed(np.random.get_state()[1][0] + worker_id)

class DataModuleFromConfig(pl.LightningDataModule):
    def __init__(self, batch_size, train=None, validation=None, test=None, predict=None,
                 wrap=False, num_workers=None, shuffle_test_loader=False, use_worker_init_fn=False,
                 shuffle_val_dataloader=False):

        super().__init__()
        self.batch_size = batch_size
        self.dataset_configs = dict()
        self.num_workers = num_workers if num_workers is not None else batch_size * 2
        self.use_worker_init_fn = use_worker_init_fn
        if train is not None:
            self.dataset_configs["train"] = train
            self.train_dataloader = self._train_dataloader
        if validation is not None:
            self.dataset_configs["validation"] = validation
            self.val_dataloader = partial(self._val_dataloader, shuffle=shuffle_val_dataloader)
        if test is not None:
            self.dataset_configs["test"] = test
            self.test_dataloader = partial(self._test_dataloader, shuffle=shuffle_test_loader)
        if predict is not None:
            self.dataset_configs["predict"] = predict
            self.predict_dataloader = self._predict_dataloader
        self.wrap = wrap

    def prepare_data(self):
        for key, data_cfg in self.dataset_configs.items():
            if key=='train':
                self.train_data=RealESRGANDataset(data_cfg)
            else:
                self.val_data=RealESRGANDataset(data_cfg)
        
        # return self.train_data,self.val_data

    def setup(self, stage=None):
        # self.datasets = dict(
        #     (k, instantiate_from_config_sr(self.dataset_configs[k]))
        #     for k in self.dataset_configs)
        
        self.datasets=dict(
            (('train',self.train_data),
            ('validation',self.val_data))
        )
        
        if self.wrap:
            for k in self.datasets:
                self.datasets[k] = WrappedDataset(self.datasets[k])
        
        return self.datasets

    def _train_dataloader(self):
        is_iterable_dataset = isinstance(self.datasets['train'], Txt2ImgIterableBaseDataset)
        if is_iterable_dataset or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None
        sampler = DistributedSampler(
                self.datasets["train"],
                num_replicas=dist.get_world_size(),
                rank=rank,
                shuffle=True,
            )   
        
        return DataLoader(self.datasets["train"], batch_size=self.batch_size,shuffle=False,sampler=sampler,pin_memory=True,
                        drop_last=True)

    def _val_dataloader(self, shuffle=False):
        if isinstance(self.datasets['validation'], Txt2ImgIterableBaseDataset) or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None
        

        sampler = DistributedSampler(
                self.datasets["validation"],
                num_replicas=dist.get_world_size(),
                rank=rank,
                shuffle=False,
            ) 
        return DataLoader(self.datasets['validation'],batch_size=self.batch_size,worker_init_fn=init_fn, shuffle=shuffle,sampler=sampler,pin_memory=True,
                        drop_last=True)

    def _test_dataloader(self, shuffle=False):
        is_iterable_dataset = isinstance(self.datasets['train'], Txt2ImgIterableBaseDataset)
        if is_iterable_dataset or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None

        # do not shuffle dataloader for iterable dataset
        shuffle = shuffle and (not is_iterable_dataset)

        return DataLoader(self.datasets["test"], batch_size=self.batch_size,
                          num_workers=self.num_workers, worker_init_fn=init_fn, shuffle=shuffle)

    def _predict_dataloader(self, shuffle=False):
        if isinstance(self.datasets['predict'], Txt2ImgIterableBaseDataset) or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None
        return DataLoader(self.datasets["predict"], batch_size=self.batch_size,
                          num_workers=self.num_workers, worker_init_fn=init_fn)

def get_parser(**parser_kwargs):
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="postfix for logdir",
    ),
    parser.add_argument(
        "-r",
        "--resume",
        type=str,
        const=True,
        default="", 
        nargs="?",
        help="resume from logdir or checkpoint in logdir",
    )
    parser.add_argument(
        "--Is_DiT",
        type=bool,
        default=True,
    )
    parser.add_argument(
        "-b",
        "--base",
        nargs="*",
        metavar="configs/bsr_sr/config_sr_finetune.yaml",
        default=list(['configs/bsr_sr/config_sr_finetune.yaml']),
        help="paths to base configs. Loaded from left-to-right. "
             "Parameters can be overwritten or added with command-line options of the form `--key value`.",
    )
    parser.add_argument(
        "-t",
        "--train",
        type=str2bool,
        const=True,
        default=True,
        nargs="?",
        help="train",
    )
    parser.add_argument(
        "--epoches",
        type=int,
        default=10000,
        help="epoch for train",
    )
    parser.add_argument(
        "--plms",
        type=bool,
        default=False,
        help="sampler scheduler",
    )
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=1.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=200,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=2,
        help="sampling batch_size",
    )
    parser.add_argument(
        '--init_img',
        type=str,
        default=""
    )
    parser.add_argument(
        "--input_size",
        type=int,
        default=128,
        help="input size",
    )
    parser.add_argument(
        "--no-test",
        type=str2bool,
        const=True,
        default=False,
        nargs="?",
        help="disable test",
    )
    parser.add_argument(
        "-p",
        "--project",
        help="name of new or path to existing project"
    )
    parser.add_argument(
        "-d",
        "--debug",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="enable post-mortem debugging",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=43,
        help="seed for seed_everything",
    )
    parser.add_argument(
        "-f",
        "--postfix",
        type=str,
        default="",
        help="post-postfix for default name",
    )
    parser.add_argument(
        "-l",
        "--logdir",
        type=str,
        default="",
        help="directory for logging dat shit",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        default="",
        help="dir to write results to",
    )
    parser.add_argument(
        "--scale_lr",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="scale base-lr by ngpu * batch_size * n_accumulate",
    )
    
    parser.add_argument('--local_rank', default=0, type=int,
                help='node rank for distributed training')
    
    return parser

def save_checkpoint(epoch,optimizer,model_folder):
    model_out_path = model_folder + "/epoch_{}.pth".format(epoch)
    # if not os.path.exists(model_folder):
    #     os.makedirs(model_folder)
    torch.save({"state_dict":model.module.state_dict(),
                "optimizer":optimizer,
                "epoch":epoch,
                },
               model_out_path)
    print("===> Checkpoint saved to {}".format(model_out_path))

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


# 计算平均损失
def average_loss_across_devices(local_loss, device):
    total_loss = torch.tensor([local_loss], dtype=torch.float32, device=device)
    dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
    total_loss /= dist.get_world_size()
    return total_loss.item()

if __name__ == "__main__":

    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    sys.path.append(os.getcwd())

    parser = get_parser()
    
    opt, unknown = parser.parse_known_args()
    batch_size = opt.n_samples
    
    configs = [OmegaConf.load(cfg) for cfg in opt.base]
    # configs=OmegaConf.load(opt.base)
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(*configs, cli)
    
    ########### DiT-model ##############
    model=LatentDiffusionSR(first_stage_config=config['first_stage_config'],
                                cond_stage_config=config['cond_stage_config'],DiT_config=config['DiT_config'],
                                **config['params'])
    
    if opt.resume:
        if not os.path.exists(opt.resume):
            raise ValueError("Cannot find {}".format(opt.resume))
        if os.path.isfile(opt.resume):
            paths = opt.resume.split("/")
            ckpt = opt.resume
            checkpoint=torch.load(ckpt,map_location='cpu')
            if opt.Is_DiT:
                model_state_dict = model.state_dict()
                for k,v in checkpoint.items():
                    if 'model.diffusion_model.'+ k in model_state_dict: 
                        model_state_dict['model.diffusion_model.'+ k]=v
                        # print("======")
            else:
                model.load_state_dict(checkpoint["state_dict"])
            print("load model")
        else:
            assert os.path.isdir(opt.resume), opt.resume

    else:
        if opt.name:
            name = "_" + opt.name
        elif opt.base:
            cfg_fname = os.path.split(opt.base[0])[-1]
            cfg_name = os.path.splitext(cfg_fname)[0]
            name = "_" + cfg_name
        else:
            name = ""
        nowname = now + name + opt.postfix
    
    logdir = opt.logdir
    ckptdir = os.path.join(logdir, "checkpoints")
    os.makedirs(ckptdir,exist_ok=True)
    
    # cfgdir = os.path.join(logdir, "configs")
    # os.makedirs(cfgdir,exist_ok=True)

    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(opt.input_size),

    ])
    
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    file_handler = logging.FileHandler(os.path.join(logdir,'log.log'))
    file_handler.setFormatter(formatter)
    
    # logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

    seed_everything(opt.seed)
        
    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir
    
    sample_path = os.path.join(outpath, "samples")
    os.makedirs(sample_path, exist_ok=True)
    input_path = os.path.join(outpath, "inputs")
    os.makedirs(input_path, exist_ok=True)
    base_count = len(os.listdir(sample_path))
    base_i = len(os.listdir(input_path))
    grid_count = len(os.listdir(outpath)) - 1
    
    ########## data ###############
    
    data=DataModuleFromConfig(**config['data'])
    data.prepare_data()
    dataset=data.setup()
    train_dataloader=data.train_dataloader()
    val_dataloader=data.val_dataloader()
    
    print("#### Data #####")
    for k in data.datasets:
        print(f"{k}, {data.datasets[k].__class__.__name__}, {len(data.datasets[k])}")
    
    model.configs=config
    
    # print(model)
    t_enc = int( opt.ddim_steps)
    niqe_list = []
    bs, base_lr = config.data.batch_size, config.model.base_learning_rate

    if torch.cuda.is_available():
        device=rank % torch.cuda.device_count()
        torch.cuda.set_device(device)
    else:
        device=torch.device('cpu')
    
    # model.to(device)
    model = DDP(model.to(device), device_ids=[rank])
    
    print("===> Setting Optimizer")
    optimizer = optim.AdamW(model.parameters(), lr=base_lr)
    lr_scheduler=CosineAnnealingLR(optimizer=optimizer,T_max=opt.epoches)
    
    print("====> Start training")
    loss_values=[]
    for epoch in tqdm(range(opt.epoches)):
        ##### train #####
        model.train()
        epoch_loss = 0.0
        for batch_idx, data in enumerate(train_dataloader):
            loss=model.module.training_step(data,logger,optimizer,batch_idx,epoch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            
            # 累加本地损失
            epoch_loss += loss.item()
        
            # 平均化损失
        average_epoch_loss = average_loss_across_devices(epoch_loss, device)
        logger.info("average_epoch_loss: %s",average_epoch_loss)
        loss_values.append(average_epoch_loss)   
        
        if epoch%1==0:
            if epoch % 5 ==0:   
                save_checkpoint(epoch=epoch,optimizer=optimizer,model_folder=ckptdir)
            img_list_old = os.listdir(opt.init_img)
            img_list_renew = []
            niqe_list = []
            psnr_list=[]
            ssim_list=[]
            all_samples = list()
            x_T = None
            save_list = os.listdir(sample_path)
            for item in img_list_old:
                img_list_renew.append(item)

            img_list = sorted(img_list_renew)
            niters = math.ceil(len(img_list) / batch_size)
            # using list comprehension
            img_list_chunk = [img_list[i * batch_size:(i + 1) * batch_size] for i in range((len(img_list) + batch_size - 1) // batch_size )]
            model.eval()
            if opt.plms:
                sampler = PLMSSampler(model.module)
            else:
                sampler = DDIMSampler(model.module)
            sampler.make_schedule(ddim_num_steps=opt.ddim_steps, ddim_eta=opt.ddim_eta, verbose=False)
            with torch.no_grad():
                
                for n in trange(niters, desc="Sampling"):
                    # seed_everything(opt.seed*n)
                    cur_img_list = img_list_chunk[n]
                    init_image_list = []
                    for item in cur_img_list:
                        cur_image = load_img(os.path.join(opt.init_img, item)).to(device)
                        cur_image = transform(cur_image)
                        init_image_list.append(cur_image)
                    init_image = torch.cat(init_image_list, dim=0)
                    
                    samples, _ = sampler.sample(t_enc, init_image.size(0), (3,opt.input_size,opt.input_size), init_image, eta=opt.ddim_eta, verbose=False, x_T=x_T)
                    x_samples = model.module.decode_first_stage(samples)
                    x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)  
                    targets = torch.clamp((init_image+1.0)/2.0, min=0.0, max=1.0)
                    for i in range(x_samples.size(0)):
                        x_sample = 255. * rearrange(x_samples[i].cpu().numpy(), 'c h w -> h w c')
                        niqe_list.append(calculate_niqe(x_sample, 0, input_order='HWC', convert_to='y'))
                        target=255. * rearrange(targets[i].cpu().numpy(), 'c h w -> h w c')
                        target =cv2.resize(target, (512,512), interpolation=cv2.INTER_LINEAR)
                        
                        psnr_list.append(compute_psnr(x_sample[:,:,0].astype(np.uint8),target[:,:,0].astype(np.uint8)))
                        ssim_list.append(compute_ssim(x_sample[:,:,0].astype(np.uint8),target[:,:,0].astype(np.uint8)))
                        
                        Image.fromarray(x_sample.astype(np.uint8)).save(
                            os.path.join(sample_path, cur_img_list[i]))
                    base_i += init_image.size(0)
                    all_samples.append(x_samples)
                
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
                grid = make_grid(grid, nrow=batch_size)

                
            assert len(niqe_list) == len(img_list)==len(psnr_list)==len(ssim_list)
            avg_niqe = np.mean(np.array(niqe_list))
            avg_npsr=np.mean(np.array(psnr_list))
            avg_ssim=np.mean(np.array(ssim_list))
            # to image
            grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
            Image.fromarray(grid.astype(np.uint8)).save(os.path.join(outpath, f'grid-{grid_count:04}-{avg_niqe}.png'))
            grid_count += 1
            logger.info(f"Average NIQE score: {avg_niqe:.3f} \n")
            logger.info(f"Average PNSR score: {avg_npsr:.3f} \n")
            logger.info(f"Average SSIM score: {avg_ssim:.3f} \n")
            print(f"Average NIQE score: {avg_niqe:.3f} \n")
            print(f"Average PNSR score: {avg_npsr:.3f} \n")
            print(f"Average SSIM score: {avg_ssim:.3f} \n")

    plt.plot(loss_values)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.savefig("Loss_curve.png")
    
                

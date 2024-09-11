import torch
import torchvision
from torch import nn, optim
from transformers import AutoTokenizer, CLIPTextModelWithProjection
# from warmup_scheduler import GradualWarmupScheduler
import numpy as np

import sys
import os
from dataclasses import dataclass

from gdf import GDF, EpsilonTarget, CosineSchedule
from gdf import VPScaler, CosineTNoiseCond, DDPMSampler, P2LossWeight, AdaptiveLossWeight

# from torchtools.transforms import SmartCrop
from torch.optim.lr_scheduler import CosineAnnealingLR

from ldm.modules.cascade.effnet import EfficientNetEncoder
from ldm.modules.cascade.stage_a import StageA

from ldm.modules.cascade.stage_b import StageB
from ldm.modules.cascade.stage_b import ResBlock, AttnBlock, TimestepBlock, FeedForwardBlock
import argparse
# from core import WarpCore
# from core.utils import EXPECTED, EXPECTED_TRAIN, load_or_fail
from omegaconf import OmegaConf
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import ModuleWrapPolicy
from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device
from contextlib import contextmanager

class WurstCore():
    # region
    # @dataclass(frozen=True)
    # class Config(TrainingCore.Config, DataCore.Config, WarpCore.Config):
    # #     # TRAINING PARAMS
    # #     lr: float = EXPECTED_TRAIN
    # #     warmup_updates: int = EXPECTED_TRAIN
    # #     shift: float = EXPECTED_TRAIN
    # #     dtype: str = None

    # #     # MODEL VERSION
    # #     model_version: str = EXPECTED  # 3BB or 700M
    # #     # clip_text_model_name: str = 'laion/CLIP-ViT-bigG-14-laion2B-39B-b160k'

    # #     # CHECKPOINT PATHS
    # #     stage_a_checkpoint_path: str = EXPECTED
    # #     effnet_checkpoint_path: str = EXPECTED
    # #     generator_checkpoint_path: str = None

    # #     # gdf customization
    # #     adaptive_loss_weight: str = None

    # @dataclass(frozen=True)
    # class Models(TrainingCore.Models, DataCore.Models, WarpCore.Models):
    #     effnet: nn.Module = EXPECTED
    #     stage_a: nn.Module = EXPECTED

    # @dataclass(frozen=True)
    # class Schedulers(WarpCore.Schedulers):
    #     generator: any = None

    # @dataclass(frozen=True)
    # class Extras(TrainingCore.Extras, DataCore.Extras, WarpCore.Extras):
    #     gdf: GDF = EXPECTED
    #     sampling_configs: dict = EXPECTED
    #     effnet_preprocess: torchvision.transforms.Compose = EXPECTED

    # info: TrainingCore.Info
    # config: Config
    # endregion
    
    def __init__(self):
        # training config
        
        
        pass
    
    
    def setup_extras_pre(self):
        gdf = GDF(
            schedule=CosineSchedule(clamp_range=[0.0001, 0.9999]),
            input_scaler=VPScaler(), target=EpsilonTarget(),
            noise_cond=CosineTNoiseCond(),
            loss_weight=AdaptiveLossWeight() if self.config.adaptive_loss_weight is True else P2LossWeight(),
        )
        sampling_configs = {"cfg": 1.5, "sampler": DDPMSampler(gdf), "shift": 1, "timesteps": 10}

        if self.info.adaptive_loss is not None:
            gdf.loss_weight.bucket_ranges = torch.tensor(self.info.adaptive_loss['bucket_ranges'])
            gdf.loss_weight.bucket_losses = torch.tensor(self.info.adaptive_loss['bucket_losses'])

        effnet_preprocess = torchvision.transforms.Compose([
            torchvision.transforms.Normalize(
                mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
            )
        ])

        transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize(self.config.image_size,
                                        interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
                                        antialias=True),
            # SmartCrop(self.config.image_size, randomize_p=0.3, randomize_q=0.2) if self.config.training else torchvision.transforms.CenterCrop(self.config.image_size)
        ])

        return self.Extras(
            gdf=gdf,
            sampling_configs=sampling_configs,
            transforms=transforms,
            effnet_preprocess=effnet_preprocess,
            clip_preprocess=None
        )

    def get_conditions(self, batch, models, extras, is_eval=False, is_unconditional=False, eval_image_embeds=False, return_fields=None):
        images = batch.get('images', None)

        if images is not None:
            images = images.to(self.device)
            if is_eval and not is_unconditional:
                effnet_embeddings = models.effnet(extras.effnet_preprocess(images))
            else:
                if is_eval:
                    effnet_factor = 1
                else:
                    effnet_factor = np.random.uniform(0.5, 1) # f64 to f32
                effnet_height, effnet_width = int(((images.size(-2)*effnet_factor)//32)*32), int(((images.size(-1)*effnet_factor)//32)*32)

                effnet_embeddings = torch.zeros(images.size(0), 16, effnet_height//32, effnet_width//32, device=self.device)
                if not is_eval:
                    effnet_images = torchvision.transforms.functional.resize(images, (effnet_height, effnet_width), interpolation=torchvision.transforms.InterpolationMode.NEAREST)
                    rand_idx = np.random.rand(len(images)) <= 0.9
                    if any(rand_idx):
                        effnet_embeddings[rand_idx] = models.effnet(extras.effnet_preprocess(effnet_images[rand_idx]))
        else:
            effnet_embeddings = None
            
        conditions = super().get_conditions(
            batch, models, extras, is_eval, is_unconditional,
            eval_image_embeds, return_fields=return_fields or ['clip_text_pooled']
        )
        # pooled_emb = conditions["clip_text_pooled"]
        # conditions["clip_text_pooled"] = torch.zeros([pooled_emb.shape[0], pooled_emb.shape[1], 1280]).to(pooled_emb.device, pooled_emb.dtype)
        return {'effnet': effnet_embeddings, 'clip': conditions['clip_text_pooled']}

    def setup_models(self, extras, skip_clip) :
        dtype = getattr(torch, self.config.dtype) if self.config.dtype else torch.float32

        # EfficientNet encoder
        effnet = EfficientNetEncoder().to(self.device)
        effnet_checkpoint = load_or_fail(self.config.effnet_checkpoint_path)
        effnet.load_state_dict(effnet_checkpoint if 'state_dict' not in effnet_checkpoint else effnet_checkpoint['state_dict'])
        effnet.eval().requires_grad_(False)
        del effnet_checkpoint

        # vqGAN
        stage_a = StageA().to(self.device)
        stage_a_checkpoint = load_or_fail(self.config.stage_a_checkpoint_path)
        stage_a.load_state_dict(stage_a_checkpoint if 'state_dict' not in stage_a_checkpoint else stage_a_checkpoint['state_dict'])
        stage_a.eval().requires_grad_(False)
        del stage_a_checkpoint

        @contextmanager
        def dummy_context():
            yield None

        loading_context = dummy_context if self.config.training else init_empty_weights

        # Diffusion models
        with loading_context():
            generator_ema = None
            if self.config.model_version == '3B':
                generator = StageB(c_hidden=[320, 640, 1280, 1280], nhead=[-1, -1, 20, 20], blocks=[[2, 6, 28, 6], [6, 28, 6, 2]], block_repeat=[[1, 1, 1, 1], [3, 3, 2, 2]])
                if self.config.ema_start_iters is not None:
                    generator_ema = StageB(c_hidden=[320, 640, 1280, 1280], nhead=[-1, -1, 20, 20], blocks=[[2, 6, 28, 6], [6, 28, 6, 2]], block_repeat=[[1, 1, 1, 1], [3, 3, 2, 2]])
            elif self.config.model_version == '700M':
                generator = StageB(c_hidden=[320, 576, 1152, 1152], nhead=[-1, 9, 18, 18], blocks=[[2, 4, 14, 4], [4, 14, 4, 2]], block_repeat=[[1, 1, 1, 1], [2, 2, 2, 2]])
                if self.config.ema_start_iters is not None:
                    generator_ema = StageB(c_hidden=[320, 576, 1152, 1152], nhead=[-1, 9, 18, 18], blocks=[[2, 4, 14, 4], [4, 14, 4, 2]], block_repeat=[[1, 1, 1, 1], [2, 2, 2, 2]])
            else:
                raise ValueError(f"Unknown model version {self.config.model_version}")

        if self.config.generator_checkpoint_path is not None:
            if loading_context is dummy_context:
                generator.load_state_dict(load_or_fail(self.config.generator_checkpoint_path))
            else:
                for param_name, param in load_or_fail(self.config.generator_checkpoint_path).items():
                    set_module_tensor_to_device(generator, param_name, "cpu", value=param)
        generator = generator.to(dtype).to(self.device)
        generator = self.load_model(generator, 'generator')

        if generator_ema is not None:
            if loading_context is dummy_context:
                generator_ema.load_state_dict(generator.state_dict())
            else:
                for param_name, param in generator.state_dict().items():
                    set_module_tensor_to_device(generator_ema, param_name, "cpu", value=param)
            generator_ema = self.load_model(generator_ema, 'generator_ema')
            generator_ema.to(dtype).to(self.device).eval().requires_grad_(False)

        if self.config.use_fsdp:
            fsdp_auto_wrap_policy = ModuleWrapPolicy([ResBlock, AttnBlock, TimestepBlock, FeedForwardBlock])
            generator = FSDP(generator, **self.fsdp_defaults, auto_wrap_policy=fsdp_auto_wrap_policy, device_id=self.device)
            if generator_ema is not None:
                generator_ema = FSDP(generator_ema, **self.fsdp_defaults, auto_wrap_policy=fsdp_auto_wrap_policy, device_id=self.device)

        if skip_clip:
            tokenizer = None
            text_model = None
        else:
            tokenizer = AutoTokenizer.from_pretrained(self.config.clip_text_model_name)
            text_model = CLIPTextModelWithProjection.from_pretrained(self.config.clip_text_model_name).requires_grad_(False).to(dtype).to(self.device)

        return self.Models(
            effnet=effnet, stage_a=stage_a,
            generator=generator, generator_ema=generator_ema,
            tokenizer=tokenizer, text_model=text_model
        )

    def setup_optimizers(self, extras, models):
        optimizer = optim.AdamW(models.generator.parameters(), lr=self.config.lr)  # , eps=1e-7, betas=(0.9, 0.95))
        optimizer = self.load_optimizer(optimizer, 'generator_optim',
                                        fsdp_model=models.generator if self.config.use_fsdp else None)
        return self.Optimizers(generator=optimizer)

    def setup_schedulers(self, extras, models,optimizers) :
        # scheduler = GradualWarmupScheduler(optimizers.generator, multiplier=1, total_epoch=self.config.warmup_updates)
        scheduler=CosineAnnealingLR(optimizers,T_max=self.info.total_steps)
        scheduler.last_epoch = self.info.total_steps
        return self.Schedulers(generator=scheduler)

    def _pyramid_noise(self, epsilon, size_range=None, levels=10, scale_mode='nearest'):
        epsilon = epsilon.clone()
        multipliers = [1]
        for i in range(1, levels):
            m = 0.75 ** i
            h, w = epsilon.size(-2) // (2 ** i), epsilon.size(-2) // (2 ** i)
            if size_range is None or (size_range[0] <= h <= size_range[1] or size_range[0] <= w <= size_range[1]):
                offset = torch.randn(epsilon.size(0), epsilon.size(1), h, w, device=self.device)
                epsilon = epsilon + torch.nn.functional.interpolate(offset, size=epsilon.shape[-2:],
                                                                    mode=scale_mode) * m
                multipliers.append(m)
            if h <= 1 or w <= 1:
                break
        epsilon = epsilon / sum([m ** 2 for m in multipliers]) ** 0.5
        # epsilon = epsilon / epsilon.std()
        return epsilon

    def forward_pass(self, data, extras, models):
        batch = next(data.iterator)

        with torch.no_grad():
            conditions = self.get_conditions(batch, models, extras)
            latents = self.encode_latents(batch, models, extras)
            epsilon = torch.randn_like(latents)
            epsilon = self._pyramid_noise(epsilon, size_range=[1, 16])
            noised, noise, target, logSNR, noise_cond, loss_weight = extras.gdf.diffuse(latents, shift=1, loss_shift=1,
                                                                                        epsilon=epsilon)

        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            pred = models.generator(noised, noise_cond, **conditions)
            loss = nn.functional.mse_loss(pred, target, reduction='none').mean(dim=[1, 2, 3])
            loss_adjusted = (loss * loss_weight).mean() / self.config.grad_accum_steps

        if isinstance(extras.gdf.loss_weight, AdaptiveLossWeight):
            extras.gdf.loss_weight.update_buckets(logSNR, loss)

        return loss, loss_adjusted

    def backward_pass(self, update, loss, loss_adjusted, models, optimizers,schedulers):
        if update:
            loss_adjusted.backward()
            grad_norm = nn.utils.clip_grad_norm_(models.generator.parameters(), 1.0)
            optimizers_dict = optimizers.to_dict()
            for k in optimizers_dict:
                if k != 'training':
                    optimizers_dict[k].step()
            schedulers_dict = schedulers.to_dict()
            for k in schedulers_dict:
                if k != 'training':
                    schedulers_dict[k].step()
            for k in optimizers_dict:
                if k != 'training':
                    optimizers_dict[k].zero_grad(set_to_none=True)
            self.info.total_steps += 1
        else:
            loss_adjusted.backward()
            grad_norm = torch.tensor(0.0).to(self.device)

        return grad_norm

    def models_to_save(self):
        return ['generator', 'generator_ema']

    def encode_latents(self, batch: dict, models, extras):
        images = batch['images'].to(self.device)
        return models.stage_a.encode(images)[0]

    def decode_latents(self, latents: torch.Tensor, batch: dict, models, extras):
        return models.stage_a.decode(latents.float()).clamp(0, 1)


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
        default="/raw7/intern/biaowang5/code/LDMSR/models/DiT-XL-2-256x256.pt", # 若是使用DiT提供的模型，则需要使用checkpoint['model']
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
        metavar="/raw7/intern/biaowang5/code/LDMSR/configs/cascade/train/finetune_b_3b.yaml",
        default=list(['/raw7/intern/biaowang5/code/LDMSR/configs/cascade/train/finetune_b_3b.yaml']),
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
        default=1000,
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
        default=8,
        help="sampling batch_size",
    )
    parser.add_argument(
        '--init_img',
        type=str,
        default="/raw7/intern/biaowang5/Data/DIV2K/DIV2K_valid_LR_bicubic/X2/"
    )
    parser.add_argument(
        "--input_size",
        type=int,
        default=512,
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
        default="./debug",
        help="directory for logging dat shit",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        default="./debug_outputs",
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
    return parser



if __name__ == '__main__':
    # print("Launching Script")
    # warpcore = WurstCore(
    #     config_file_path=sys.argv[1] if len(sys.argv) > 1 else None,
    #     device=torch.device(int(os.environ.get("SLURM_LOCALID")))
    # )
    # # core.fsdp_defaults['sharding_strategy'] = ShardingStrategy.NO_SHARD
    # # RUN TRAINING
    # warpcore()
    
    parser = get_parser()
    opt, unknown = parser.parse_known_args()
    
    batch_size = opt.n_samples
    
    configs = [OmegaConf.load(cfg) for cfg in opt.base]
    # configs=OmegaConf.load(opt.base)
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(*configs, cli)

    
    core=WurstCore()
    
    extras = core.setup_extras_pre()
    data = core.setup_data(extras)
    models = core.setup_models(extras)
    models.generator.bfloat16()
    print("STAGE B READY")
    
    
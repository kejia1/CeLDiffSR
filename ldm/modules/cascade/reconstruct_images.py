import os
import yaml
import torch
from tqdm import tqdm

# os.chdir('..')
# from inference.utils import *
# from core.utils import load_or_fail
from train import WurstCoreB

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


### Load Config.

# SETUP STAGE B & A
config_file_b = 'configs/inference/stage_b_3b.yaml'
with open(config_file_b, "r", encoding="utf-8") as file:
    config_file_b = yaml.safe_load(file)
    
core = WurstCoreB(config_dict=config_file_b, device=device, training=False)

### Load Extras, Data & Models

extras = core.setup_extras_pre()
data = core.setup_data(extras)
models = core.setup_models(extras)
models.generator.bfloat16()
print("STAGE B READY")

### Preview Batch

batch = next(data.iterator)
print("ORIG SIZE:", batch['images'].shape)

show_images(batch['images'])

print(batch['captions'])

## Step-by-Step Explanation
""""
A quick reminder, StableCascade uses three Stages (A, B & C). 
Stage C is responsible for the text-conditional generation in the highly compressed latent space. 
On the other hand, Stage A & B are used for achieving this high compression, thus encoding and decoding images to the latent space. 
Specific details can be found in the [paper](https://openreview.net/forum?id=gU58d5QeGv), 
but you only need to know that Stage A is just a VAE, providing a small compression factor of 4. (`4 x 3 x 1024 x 1024 -> 4 x 4 x 256 x 256`). 
Then Stage B is learnt on top of that compressed latent space, to compress images even further. 
This cascading would not be possible when just using another VAE. 
Therefore, a more powerful approach is needed: a diffusion model.
Stage B iteratively reconstructs images into the latent space of the VAE (Stage A), 
from where it can be decoded into the pixel-space.
Let's set the sampling parameters for Stage B:
"""
extras.sampling_configs['cfg'] = 1.1
extras.sampling_configs['shift'] = 1
extras.sampling_configs['timesteps'] = 10
extras.sampling_configs['t_start'] = 1.0

""""
Next we encode the images. 
By default, the encoder (an EfficientNet architecture) yields feature representations with a compression factor of 32. 
This would mean that we encode our images like this: `4 x 3 x 1024 x 1024 -> 4 x 16 x 32 x 32`. 
To increase this even further, during training we additionally downscale images before encoding them. 
There might be other ways of achieving encoding images even further (e.g. adding more layers), but this works as well. 

During training we use a downscaling factor randomly between 1.0 and 0.5, yielding compression factors between 32 and 64. 
That means Stage B, the diffusion model, can compress images of size `4 x 3 x 1024 x 1024` to latent dimensions between `4 x 16 x 32 x 32` and `4 x 16 x 16 x 16`. 
So up to a spatial compression factor of 64! Of course, the more you compress, the more details you lose in the reconstructions. 
We found that a downscaling factor of 0.75 works very well and preserves a lot of details, resulting in a spatial compression factor of `42`.

In the code below, you can test different downscaling factors.
"""
print("Original Size:", batch['images'].shape)
factor = 3/4
scaled_image = downscale_images(batch['images'], factor)
print("[Optional] Downscaled Size:", scaled_image.shape)

effnet_latents = models.effnet(extras.effnet_preprocess(scaled_image.to(device)))
print("Encoded Size:", effnet_latents.shape)

""""
Now, we set the conditions for the diffusion model (Stage B). We condition the model on text as well, 
however the effect of it is tiny, especially when the `effnet_latents` are given as well, 

because they are just so powerful.
"""
conditions = core.get_conditions(batch, models, extras, is_eval=True, is_unconditional=False)
unconditions = core.get_conditions(batch, models, extras, is_eval=True, is_unconditional=True)    
conditions['effnet'] = effnet_latents
unconditions['effnet'] = torch.zeros_like(effnet_latents)


with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
    sampling_b = extras.gdf.sample(
        models.generator, conditions, (batch['images'].size(0), 4, batch['images'].size(-2)//4, batch['images'].size(-1)//4),
        unconditions, device=device, **extras.sampling_configs
    )
    for (sampled_b, _, _) in tqdm(sampling_b, total=extras.sampling_configs['timesteps']):
        sampled_b = sampled_b
    sampled = models.stage_a.decode(sampled_b).float()
    print("Decoded Size:", sampled.shape)
    
    
show_images(batch['images'])
show_images(sampled)




# Stage B Parameters
extras.sampling_configs['cfg'] = 1.1
extras.sampling_configs['shift'] = 1
extras.sampling_configs['timesteps'] = 10
extras.sampling_configs['t_start'] = 1.0

with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
    # torch.manual_seed(42)

    print("Original Size:", batch['images'].shape)
    factor = 3/4
    scaled_image = downscale_images(batch['images'], factor)
    print("[Optional] Downscaled Size:", scaled_image.shape)
    
    effnet_latents = models.effnet(extras.effnet_preprocess(scaled_image.to(device)))
    print("Encoded Size:", effnet_latents.shape)
    
    conditions = core.get_conditions(batch, models, extras, is_eval=True, is_unconditional=False)
    unconditions = core.get_conditions(batch, models, extras, is_eval=True, is_unconditional=True)    
    conditions['effnet'] = effnet_latents
    unconditions['effnet'] = torch.zeros_like(effnet_latents)

    sampling_b = extras.gdf.sample(
        models.generator, conditions, (batch['images'].size(0), 4, batch['images'].size(-2)//4, batch['images'].size(-1)//4),
        unconditions, device=device, **extras.sampling_configs
    )
    for (sampled_b, _, _) in tqdm(sampling_b, total=extras.sampling_configs['timesteps']):
        sampled_b = sampled_b
    sampled = models.stage_a.decode(sampled_b).float()
    print("Decoded Size:", sampled.shape)
        
show_images(batch['images'])
show_images(sampled)
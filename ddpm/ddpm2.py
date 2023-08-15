from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
import torch, gc 
from torch import nn


torch.cuda.set_device('cuda:0')




# Create a new model object
model = Unet(
    dim = 32,
    dim_mults = (1, 2, 4, 8),
    flash_attn = True
)

# Now, you can access attributes directly
#print(model.channels)  # This will print the 'channels' attribute of the original model

# Same for the diffusion
diffusion = GaussianDiffusion(model, image_size=256, timesteps=1000, sampling_timesteps=250)






trainer = Trainer(
    diffusion,
    '/data/gongmo/syntheticData_AI_Detector/non_synthetic/Normal',
    train_batch_size = 8,
    train_lr = 8e-5,
    train_num_steps = 10000,         # total training steps
    gradient_accumulate_every = 2,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    amp = True,                       # turn on mixed precision
    calculate_fid = True,
    convert_image_to='RGB'# whether to calculate fid during training
             # Convert input images to RGB
)


trainer.train()




# Create a batch of random noise
num_images = 1000
noise_shape = (num_images, 3, 256, 256)
noise = torch.randn(*noise_shape).to(diffusion.device)

# Generate images
with torch.no_grad(): 
    generated_images = diffusion.ddim_sample(noise_shape)


from PIL import Image
import numpy as np
import os

# Make sure the directory exists
os.makedirs('/data/gongmo/team1/gongmo_2023/ddpm/data_timestep10000', exist_ok=True)

for i, image in enumerate(generated_images):
    # Convert to numpy array and denormalize
    image = image.permute(1, 2, 0).cpu().numpy()
    image = ((image + 1) / 2 * 255).astype(np.uint8)

    # Create a PIL Image and save it
    pil_image = Image.fromarray(image)
    pil_image.save(f'/data/gongmo/team1/gongmo_2023/ddpm/data/generated_image_{i}.png')

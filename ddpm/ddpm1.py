from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
import torch, gc 
from torch import nn

device_ids = [0, 1, 2, 3]
output_device = device_ids[0]


torch.cuda.set_device(output_device)
torch.cuda.empty_cache()
gc.collect()



class CustomDataParallel(nn.DataParallel):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)

# Create a new model object
model = Unet(
    dim = 64,
    dim_mults = (1, 2, 4, 8),
    flash_attn = True
)

# Make the model parallel
model = CustomDataParallel(model, device_ids=device_ids, output_device=output_device)

model = model.to(output_device)
# Now, you can access attributes directly
#print(model.channels)  # This will print the 'channels' attribute of the original model

# Same for the diffusion
diffusion = GaussianDiffusion(model, image_size=256, timesteps=1000, sampling_timesteps=250)
diffusion = CustomDataParallel(diffusion, device_ids=device_ids, output_device=output_device)
diffusion = diffusion.to(output_device)






trainer = Trainer(
    diffusion.module,
    '/data/gongmo/syntheticData_AI_Detector/non_synthetic/Normal',
    train_batch_size = 8,
    train_lr = 8e-5,
    train_num_steps = 1,         # total training steps
    gradient_accumulate_every = 2,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    amp = True,                       # turn on mixed precision
    calculate_fid = True,
    convert_image_to='RGB'# whether to calculate fid during training
             # Convert input images to RGB
)


trainer.train()




# Create a batch of random noise
num_images = 16  
noise = torch.randn(num_images, 3, 254, 254).to(diffusion.device)  # 변경된 부분

# Generate images
with torch.no_grad(): 
    generated_images = diffusion.ddim_sample(noise, timesteps=250)


from PIL import Image
import numpy as np
import os

# Make sure the directory exists
os.makedirs('/data/gongmo/team1/gongmo_2023/ddpm/data', exist_ok=True)

for i, image in enumerate(generated_images):
    # Convert to numpy array and denormalize
    image = image.permute(1, 2, 0).cpu().numpy()
    image = ((image + 1) / 2 * 255).astype(np.uint8)

    # Create a PIL Image and save it
    pil_image = Image.fromarray(image)
    pil_image.save(f'/data/gongmo/team1/gongmo_2023/ddpm/data/generated_image_{i}.png')

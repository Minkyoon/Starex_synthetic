from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
import torch, gc 
from torch import nn


torch.cuda.set_device('cuda:0')




# Create a new model object
model = Unet(
    dim = 64,
    dim_mults = (1, 2, 4, 8),
    flash_attn = True
)

# Now, you can access attributes directly
#print(model.channels)  # This will print the 'channels' attribute of the original model

# Same for the diffusion
diffusion = GaussianDiffusion(model, image_size=128, timesteps=1000, sampling_timesteps=250)






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

#Save the model
model_path = "/data/gongmo/team1/gongmo_2023/ddpm/model_128_10000.pth"
torch.save(diffusion.model.state_dict(), model_path)

# Load the model
model_path = "/data/gongmo/team1/gongmo_2023/ddpm/model_128_10000.pth"
model = Unet(dim=64, dim_mults=(1, 2, 4, 8), flash_attn=True)
model.load_state_dict(torch.load(model_path))
model.to(diffusion.device)  # Ensure the model is on the correct device

# You can now use the model for inference
# e.g., generated_images = diffusion.sample(batch_size=1)



# Create a batch of random noise


from PIL import Image
import numpy as np
import os

# Make sure the directory exists
file_path='/data/gongmo/team1/gongmo_2023/ddpm/data_128'
os.makedirs(file_path, exist_ok=True)
with torch.no_grad(): 
    for i in range(1000):
        generated_image = diffusion.sample(batch_size=1) # Generate one image at a time

        # Convert to numpy array and denormalize
        image = generated_image[0].permute(1, 2, 0).cpu().numpy()
        image = ((image + 1) / 2 * 255).astype(np.uint8)

        # Create a PIL Image and save it
        pil_image = Image.fromarray(image)
        pil_image.save(f'{file_path}/generated_image_{i}.png')

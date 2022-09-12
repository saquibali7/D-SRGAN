import os
import wandb
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset , DataLoader
from glob import glob
import torchvision.transforms as transforms
from generator import Generator
from discriminator import Discriminator
from utils import calculate_error, TV_loss
from torchvision.utils import make_grid
from dataset import train_loader, test_loader
from tqdm import tqdm
import wandb
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

generator = Generator(1, 128)
generator = generator.to(device)
discriminator = Discriminator(1, 128)
discriminator = discriminator.to(device)

optim_G = torch.optim.Adam(generator.parameters(), lr=0.0001)
optim_D = torch.optim.Adam(discriminator.parameters(), lr=0.0001)

wandb.init(project="sr-gan")
num_epochs = 250
num_train_batches = float(len(train_loader))
num_val_batches = float(len(test_loader))

for epoch in range(num_epochs):
    print(f"Epoch {epoch}: ", end ="")
    
    G_adv_loss = 0
    G_rec_loss = 0
    G_tot_loss = 0
    D_adv_loss = 0
    
    generator.train()
    for batch, (lr, hr) in enumerate(train_loader):

      for p in discriminator.parameters():
        p.requires_grad = False
        #training generator
      optim_G.zero_grad()
 
      lr_images = lr.to(device)
      hr_images = hr.to(device)
      lr_images = lr_images.float()
      predicted_hr_images = generator(lr_images)
      predicted_hr_labels = discriminator(predicted_hr_images)
      gf_loss = F.binary_cross_entropy_with_logits(predicted_hr_labels, torch.ones_like(predicted_hr_labels)) #adverserial loss

      # reconstruction loss

      # gr_loss = 100*F.l1_loss(predicted_hr_images, hr_images) # L1 loss
      tv_loss = TV_loss(predicted_hr_images,0.0000005)
      gr_loss = 100*F.mse_loss(predicted_hr_images, hr_images) + tv_loss # L2 loss

      g_loss = gf_loss + gr_loss 

      G_adv_loss += gf_loss.item()
      G_rec_loss += gr_loss.item()
      G_tot_loss += g_loss.item()
      
      g_loss.backward()
      optim_G.step()
      
      # training discriminator
      for p in discriminator.parameters():
        p.requires_grad = True
      optim_D.zero_grad()
      predicted_hr_images = generator(lr_images).detach() # avoid back propogation to generator
      hr_images = hr_images.float()
      adv_hr_real = discriminator(hr_images)
      adv_hr_fake = discriminator(predicted_hr_images)
      df_loss = F.binary_cross_entropy_with_logits(adv_hr_real, torch.ones_like(adv_hr_real)) + F.binary_cross_entropy_with_logits(adv_hr_fake, torch.zeros_like(adv_hr_fake))
      D_adv_loss += df_loss.item()
      df_loss.backward()
      optim_D.step()
    
    wandb.log({"G Adversarial Loss": G_adv_loss/num_train_batches, 'epoch':epoch })
    wandb.log({"G Reconstruction Loss": G_rec_loss/num_train_batches, 'epoch':epoch })
    wandb.log({"G Loss Total": G_tot_loss/num_train_batches, 'epoch':epoch })
    wandb.log({"D Adversarial Loss": D_adv_loss/num_train_batches, 'epoch':epoch })


    #After each epoch, we perform validation
    with torch.inference_mode():
      val_psnr = 0
      val_ssim = 0
      for batch_idx, (lr, hr) in enumerate(train_loader):
        lr = lr.to(device)
        hr = hr.to(device)
        lr = lr.float()
        predicted_hr = generator(lr)

        psnr, ssim = calculate_error(hr, predicted_hr)
        val_psnr += psnr
        val_ssim += ssim

        grid1 = make_grid(lr)
        grid2 = make_grid(hr)
        grid3 = make_grid(predicted_hr)
        grid1 = wandb.Image(grid1, caption="Low Resolution Image")
        grid2 = wandb.Image(grid2, caption="High Resolution Image")
        grid3 = wandb.Image(grid3, caption="Reconstructed High Resolution Image")
        wandb.log({"Original LR": grid1})
        wandb.log({"Original HR": grid2})
        wandb.log({"Reconstruced": grid3})
    
    val_psnr /= num_val_batches
    val_ssim /= num_val_batches
    wandb.log({"PSNR" : val_psnr, 'epoch':epoch })
    wandb.log({"SSIM" : val_ssim, 'epoch':epoch })
    print(f"PSNR: {val_psnr:.3f} SSIM: {val_ssim:.3f}\n")

import torch
import torch.nn as nn
import torch.autograd as autograd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.model.prnu_generator import PRNUGenerator, PRNUCritic

BATCH_SIZE = 8
LR = 0.0001
B1 = 0.0
B2 = 0.9
N_CRITIC = 5
LAMBDA_GP = 10
EPOCHS = 1000
LATENT_DIM = 128
PRNU_SCALE = 1000.0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- DATASET LOADER ---
class PRNUDataset(Dataset):
    def __init__(self, data_dir, crop_size=512):
        self.files = list(Path(data_dir).rglob("*_fingerprint.npy"))
        self.crop_size = crop_size
        if len(self.files) == 0:
            raise ValueError("No .npy files found! Check your paths.")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # Load the raw math
        noise = np.load(self.files[idx]).astype(np.float32)
        
        # Scale the microscopic noise up so Tanh() can actually process it
        noise = noise * PRNU_SCALE
        
        # Clip to ensure it strictly stays within [-1, 1] to match Generator Tanh()
        noise = np.clip(noise, -1.0, 1.0)
        
        # Convert to tensor format: shape [C, H, W]
        if noise.ndim == 2:
            noise = np.expand_dims(noise, axis=0)
            noise = np.repeat(noise, 3, axis=0) # Fake 3 channels if grayscale
        elif noise.shape[2] == 3:
            noise = np.transpose(noise, (2, 0, 1)) # HWC to CHW
            
        # --- FORENSIC CROPPING ---
        _, h, w = noise.shape
        
        if h < self.crop_size or w < self.crop_size:
            raise RuntimeError(f"Noise array {self.files[idx]} is smaller than crop size {self.crop_size}.")
            
        # Randomly select a 512x512 patch
        top = np.random.randint(0, h - self.crop_size + 1)
        left = np.random.randint(0, w - self.crop_size + 1)
        
        noise_crop = noise[:, top:top+self.crop_size, left:left+self.crop_size]
            
        return torch.tensor(noise_crop)

# --- GRADIENT PENALTY ---
def compute_gradient_penalty(critic, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = torch.rand((real_samples.size(0), 1, 1, 1), device=DEVICE)
    
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = critic(interpolates)
    
    fake = torch.ones((real_samples.size(0), 1), device=DEVICE)
    
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

# --- TRAINING LOOP ---
def train():
    print(f"Training on {DEVICE}...")
    
    dataset = PRNUDataset("../datasets/prnu_fingerprints")
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    
    generator = PRNUGenerator(latent_dim=LATENT_DIM).to(DEVICE)
    critic = PRNUCritic().to(DEVICE)
    
    # WGAN-GP uses Adam but with specific momentum bounds
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=LR, betas=(B1, B2))
    optimizer_C = torch.optim.Adam(critic.parameters(), lr=LR, betas=(B1, B2))
    
    for epoch in range(EPOCHS):
        for i, real_imgs in enumerate(dataloader):
            real_imgs = real_imgs.to(DEVICE)
            
            # ---------------------
            #  Train Critic
            # ---------------------
            optimizer_C.zero_grad()
            
            # Generate a batch of fake noise
            z = torch.randn(BATCH_SIZE, LATENT_DIM, device=DEVICE)
            fake_imgs = generator(z)
            
            # Real images evaluate to positive scores, fake to negative
            real_validity = critic(real_imgs)
            fake_validity = critic(fake_imgs.detach())
            
            # Gradient penalty
            gradient_penalty = compute_gradient_penalty(critic, real_imgs.data, fake_imgs.data)
            
            # Adversarial loss
            d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + LAMBDA_GP * gradient_penalty
            
            d_loss.backward()
            optimizer_C.step()
            
            # ---------------------
            #  Train Generator
            # ---------------------
            if i % N_CRITIC == 0:
                optimizer_G.zero_grad()
                
                # Generate a batch of fake images
                fake_imgs = generator(z)
                
                # Loss measures generator's ability to fool the critic
                fake_validity = critic(fake_imgs)
                g_loss = -torch.mean(fake_validity)
                
                g_loss.backward()
                optimizer_G.step()
                
                print(
                    f"[Epoch {epoch}/{EPOCHS}] [Batch {i}/{len(dataloader)}] "
                    f"[D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}]"
                )

    print("\nTraining complete. Saving collapsed model for team autopsy...")
    save_path = "synthetic_prnu_generator.pth"
    torch.save(generator.state_dict(), save_path)
    print(f"Garbage weights successfully saved to {save_path}")

if __name__ == "__main__":
    train()
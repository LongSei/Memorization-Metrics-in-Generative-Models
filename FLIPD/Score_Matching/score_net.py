from torchcfm.models.unet import UNetModel
import torch
from Score_Matching.sde import *
from torch.optim import Adam
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm

class ScoreNet(): 
    def __init__(self, 
                image_dim=(1, 28, 28),
                use_condition=False, 
                num_classes=10,
                device='cuda'): 
        
        if use_condition == False: 
            self.model = UNetModel(
                dim=image_dim,
                num_channels=32, 
                num_res_blocks=1,
                class_cond=False
            ).to(device)
        else: 
            self.model = UNetModel(
                dim=image_dim,
                num_channels=32, 
                num_res_blocks=1,
                class_cond=True, 
                num_classes=num_classes
            ).to(device)
            
        self.sde = subVPSDE(
            beta_min=0.1, 
            beta_max=20, 
            discretization_steps=1000
        )
        
        self.device = device
        self.epsilon = 1e-6
        self.image_dim = image_dim
        self.num_classes = num_classes
        
    def loss_fn(self,x,y): 
        t = torch.rand(x.shape[0], device=self.device) * (1. - self.epsilon) + self.epsilon
        z = torch.randn_like(x, requires_grad=True)
        _, std = self.sde.marginal_probability(x, t)
        perturbed_x = x + z * std[:, None, None, None]
        score = self.model(t, perturbed_x, y)
        loss = torch.mean(torch.sum((score * std[:, None, None, None] + z)**2, dim=(1,2,3)))
        return loss

    def train(self,
              data_loader,
              optimizer,
              num_epochs=100,
              device='cuda'):
        """
        Train the model
        """
        self.model.to(device)
        
        for epoch in tqdm(range(num_epochs), ncols=88):
            avg_loss = 0.
            num_items = 0
            for x, y in data_loader:
                x = x.to(self.device)
                y = y.to(self.device)
                loss = self.loss_fn(x,y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                avg_loss += loss.item() * x.shape[0]
                num_items += x.shape[0]
            tqdm.write('Average Loss: {:5f}'.format(avg_loss / num_items))

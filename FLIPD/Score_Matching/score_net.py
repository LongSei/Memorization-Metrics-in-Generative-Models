from torchcfm.models.unet import UNetModel
import torch
from Score_Matching.sde import DDPM
from torch.optim import Adam
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm

class ScoreNet(): 
    def __init__(self, 
                image_dim=(1, 28, 28),
                use_condition=False, 
                num_classes=10,
                device='cpu'): 
        
        if use_condition: 
            self.model = UNetModel(
                dim=image_dim,
                num_channels=32, 
                num_res_blocks=1,
                class_cond=False
            )
        else: 
            self.model = UNetModel(
                dim=image_dim,
                num_channels=32, 
                num_res_blocks=1,
                class_cond=True, 
                num_classes=num_classes
            )
            
        self.sde = DDPM(
            beta_min=0.1, 
            beta_max=20, 
            discretization_steps=1000
        )
        
        self.device = device
        self.epsilon = 1e-6
        
    def loss_fn(self,x): 
        t = (torch.rand(1, device=x.device) + torch.arange(len(x), device=x.device) / len(x)) % (1 - self.epsilon)
        t = t[:, None]
        z = torch.randn_like(x, requires_grad=True)
        _, std = self.sde.marginal_probability(x, t)
        perturbed_x = x + torch.multiply(z, std)
        score = self.model(t, perturbed_x)
        loss = torch.mean(torch.sum((score * std[:, None, None, None] + z)**2, dim=(1,2,3)))
        return loss
    
    def train(self,
              data_loader,
              optimizer,
              num_epochs=100,
              device='cpu'):
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
                loss = self.loss_fn(x)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                avg_loss += loss.item() * x.shape[0]
                num_items += x.shape[0]
            tqdm.write('Average Loss: {:5f}'.format(avg_loss / num_items))
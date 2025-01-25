from diffusers import DDIMScheduler, StableDiffusionPipeline 
from tqdm import tqdm
import gc
import torch
import matplotlib.pyplot as plt 
import numpy as np

class DiffusionPipelineUtils(): 
    def __init__(self, 
                pipeline: StableDiffusionPipeline, 
                num_images_per_prompt: int=1, 
                do_classifier_free_guidance: bool=True,
                device='cpu'):
        """
        Constructor for DiffusionPipelineUtils

        Args:
            pipeline (StableDiffusionPipeline): pipeline
            num_images_per_prompt (int, optional): amount of images per prompt. Defaults to 1.
            do_classifier_free_guidance (bool, optional): whether to do classifier free guidance. Defaults to True.
            device (str, optional): device. Defaults to 'cpu'.
        """
        self.pipeline = pipeline
        self.device = device 
        self.num_images_per_prompt = num_images_per_prompt
        self.do_classifier_free_guidance = do_classifier_free_guidance
        
    def set_nums_images_per_prompt(self, num_images_per_prompt): 
        self.num_images_per_prompt = num_images_per_prompt
        
    def set_device(self, device): 
        self.device = device
    
    def set_do_classifier_free_guidance(self, do_classifier_free_guidance): 
        self.do_classifier_free_guidance = do_classifier_free_guidance

    def __prompt_embedding(self, prompt: str): 
        """
        Embed the prompt using the prompt encoder
        
        Args: 
            prompt (str): prompt to embed
            
        Returns: 
            unconditional_embedding (torch.Tensor)
            conditional_embedding (torch.Tensor)
        """
        prompt_embedding_tuple = self.pipeline.encode_prompt(
            prompt=prompt,
            device=self.device,
            num_images_per_prompt=self.num_images_per_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance
        )
        
        unconditional_embedding = prompt_embedding_tuple[0]
        conditional_embedding = prompt_embedding_tuple[1]

        return unconditional_embedding, conditional_embedding
    
    @torch.no_grad()
    def get_mean_and_std(self, t, x):
        # Get the scheduler and extract noise schedule
        betas = self.pipeline.scheduler.betas  # Noise schedule

        # Compute \(\bar{\alpha}_t\)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0).to(self.device)  # \(\bar{\alpha}_t\)

        # Compute mean and variance
        alpha_bar_t = alphas_cumprod[t.int()]
        mean_t = torch.sqrt(alpha_bar_t) * x
        variance_t = 1.0 - alpha_bar_t
        
        return mean_t, variance_t
    
    @torch.no_grad()
    def prepare_latents(self, 
                        t, latents: torch.Tensor):
        """
        Prepare latents for the diffusion process

        Args:
            t (torch.Tensor): time step
            latents (torch.Tensor): latents for the diffusion process

        Returns:
            latent_model_input (torch.Tensor): latents for the model input
        """
        latent_model_input = (
            torch.cat([latents, latents])
        )
        latent_model_input = self.pipeline.scheduler.scale_model_input(
            latent_model_input, 
            t
        )

        return latent_model_input
    
    @torch.no_grad()
    def prepare_prompt_input(self, prompt: str): 
        prompt_embeds = self.__prompt_embedding(prompt=prompt)

        single_prompt_embeds = prompt_embeds[0][:, :].clone().detach()
        single_prompt_embeds.requires_grad = True
        
        dummy_prompt_embeds = prompt_embeds[-1][:, :].clone()
        
        input_prompt_embeds = torch.cat(
            [
                dummy_prompt_embeds.repeat(self.num_images_per_prompt, 1, 1),
                single_prompt_embeds.repeat(self.num_images_per_prompt, 1, 1),
            ]
        )
        
        return input_prompt_embeds
    
    @torch.no_grad()
    def get_noise_predict(self, 
                        prompt: str,
                        latent_model_input,
                        t):
        
        input_prompt_embeds = self.prepare_prompt_input(
            prompt=prompt
        )
        
        noise_predict = self.pipeline.unet(
            latent_model_input, 
            t, 
            encoder_hidden_states=input_prompt_embeds, 
            cross_attention_kwargs=None, 
            return_dict=False
        )[0]
        
        return noise_predict
    
    def score_fn(self, t, x, guidance_scale: int, prompt: str): 
        prompt_embeds = self.__prompt_embedding(prompt=prompt)

        single_prompt_embeds = prompt_embeds[0][:, :].clone().detach()
        single_prompt_embeds.requires_grad = True
        
        dummy_prompt_embeds = prompt_embeds[-1][:, :].clone()
        
        noise_uncond, noise_pred_text = self.pipeline.unet(
            torch.cat([x, x]),
            t,
            encoder_hidden_states=torch.cat([dummy_prompt_embeds.repeat(x.shape[0], 1, 1),single_prompt_embeds.repeat(x.shape[0], 1, 1)]),
            cross_attention_kwargs=None,
            return_dict=False,
        )[0].chunk(2)
        
        return noise_uncond + guidance_scale * (noise_pred_text - noise_uncond)

    @torch.no_grad()
    def sampling(self, 
                prompt: str, 
                num_inference_steps: int=100,
                guidance_scale: int=3.5,
                use_formula: bool=True): 
        """
        Perform sampling with epsilon formula

        Args:
            prompt (str): prompt
            num_inference_steps (int, optional): amount of inference steps. Defaults to 100.
            guidance_scale (int, optional): guidance scale . Defaults to 3.5.
            use_formula (bool, optional): whether to use the formula. Defaults to True.
            
        Returns:
            samples (torch.Tensor): samples
        """
        
        self.pipeline.scheduler.set_timesteps(num_inference_steps, device=self.device)
        
        start_latents = torch.randn(1, 4, 64, 64, device=self.device)
        start_latents *= self.pipeline.scheduler.init_noise_sigma
        latents = start_latents.clone()
        
        for idx in tqdm(range(num_inference_steps)): 
            t = self.pipeline.scheduler.timesteps[idx]
            
            latent_model_input = self.prepare_latents(
                t, latents
            )
            
            noise_predict = self.get_noise_predict(
                prompt=prompt,
                latent_model_input=latent_model_input,
                t=t
            )
            
            if self.do_classifier_free_guidance is True: 
                noise_predict_uncond, noise_predict_text = noise_predict.chunk(2)
                noise_predict = noise_predict_uncond + guidance_scale * (noise_predict_text - noise_predict_uncond)
            
            if use_formula is True: 
                prev_t = max(1, t.item() - (1000 // num_inference_steps))  # t-1
                alpha_t = self.pipeline.scheduler.alphas_cumprod[t.item()]
                alpha_t_prev = self.pipeline.scheduler.alphas_cumprod[prev_t]
                predicted_x0 = (latents - (1 - alpha_t).sqrt() * noise_predict) / alpha_t.sqrt()
                direction_pointing_to_xt = (1 - alpha_t_prev).sqrt() * noise_predict
                latents = alpha_t_prev.sqrt() * predicted_x0 + direction_pointing_to_xt
            else: 
                latents = self.pipeline.scheduler.step(noise_predict, t, latents).prev_sample
            
        images = self.pipeline.decode_latents(latents)
        images = self.pipeline.numpy_to_pil(images)

        return images
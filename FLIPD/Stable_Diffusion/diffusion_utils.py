from diffusers import DDIMScheduler, StableDiffusionPipeline 
import torch

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
            prompt (str): prompt 
            num_images_per_prompt (int): amount of image for each prompt
            do_classifier_free_guidance (bool): do classifier free guidance
            
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
    
    def prepare_latents(self, 
                        t, 
                        batch_size: int,
                        num_channels_latents: int,
                        latent_height: int,
                        latent_width: int,
                        latent_dtype: torch.dtype=torch.float32):
        """
        Prepare latents for the diffusion process

        Args:
            batch_size (int): batch size
            num_images_per_prompt (int): number of images per prompt
            num_channels_latents (int): number of channels of the latents
            latent_height (int): height of the latent
            latent_width (int): width of the latent
            latent_dtype (torch.dtype): data type of the latent

        Returns:
            latent_model_input (torch.Tensor): latents for the model input
        """
        latents = self.pipeline.prepare_latents(
            batch_size * self.num_images_per_prompt,
            num_channels_latents,
            latent_height,
            latent_width,
            latent_dtype,
            self.device,
            None,
            None
        )

        latent_model_input = (
            torch.cat([latents, latents])
        )
        latent_model_input = self.pipeline.scheduler.scale_model_input(
            latent_model_input, 
            t
        )

        return latent_model_input
    
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
    
    def get_noise_predict(self, 
                        prompt: str,
                        t, 
                        batch_size: int,
                        num_channels_latents: int,
                        latent_height: int,
                        latent_width: int,
                        latent_dtype: torch.dtype=torch.float32):
        latent_model_input = self.prepare_latents(
            t, batch_size=batch_size, 
            num_channels_latents=num_channels_latents, 
            latent_height=latent_height, 
            latent_width=latent_width, 
            latent_dtype=latent_dtype
        )
        
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
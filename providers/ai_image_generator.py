import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler, AutoencoderKL
from PIL import Image
from typing import List, Optional
from pathlib import Path
from get_prompts import get_negative_prompt, get_image_guide_prompt
from .image_provider import ImageProvider

class AIImageGenerator(ImageProvider):
    """Generates images using AI models like Stable Diffusion"""
    
    def __init__(self, save_dir: str = "uploads/generated_image", device: str = "cuda", **kwargs):
        """
        Initialize the AI Image Generator with the specified model and settings
        
        Args:
            save_dir: Directory to save generated images
            device: Device to run inference on ("cuda" or "cpu")
        """
        super().__init__(save_dir)
        self.device = device
        
    def _initialize_pipeline(self):
        """Initialize the Stable Diffusion pipeline with optimized settings"""
        try:
            # Load the VAE separately for memory efficiency
            vae = AutoencoderKL.from_pretrained(self.vae_id, torch_dtype=self.dtype)
            
            # Load the pipeline with memory-efficient settings
            self.pipeline = StableDiffusionPipeline.from_pretrained(
                self.model_id,
                vae=vae,
                torch_dtype=self.dtype,
                use_safetensors=True,
                low_cpu_mem_usage=True,
            )
            self.pipeline = self.pipeline.to(self.device)
            
            # Memory optimizations
            self.pipeline.enable_attention_slicing(slice_size="auto")
            self.pipeline.enable_vae_tiling()
            self.pipeline.enable_model_cpu_offload()
            
            # Use an efficient scheduler
            self.pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
                self.pipeline.scheduler.config, 
                algorithm_type="sde-dpmsolver++", 
                use_karras_sigmas=True
            )
            
            # For better prompt handling
            self.max_token_length = 75  # Safe limit for standard SD models
            
        except Exception as e:
            print(f"Error initializing AI Image Generator: {e}")
            raise
    
    def truncate_prompt(self, prompt: str) -> str:
        """Truncate prompt to a safe length for the tokenizer"""
        tokens = self.pipeline.tokenizer.tokenize(prompt)
        if len(tokens) > self.max_token_length:
            truncated_tokens = tokens[:self.max_token_length]
            truncated_prompt = self.pipeline.tokenizer.convert_tokens_to_string(truncated_tokens)
            print(f"Prompt truncated from {len(tokens)} to {len(truncated_tokens)} tokens")
            return truncated_prompt
        return prompt
    
    def generate_image(self, 
                      prompt: str,
                      negative_prompt: Optional[str] = None,
                      num_inference_steps: int = 30,
                      guidance_scale: float = 7.0,
                      height: int = 640,
                      width: int = 640) -> Image.Image:
        """
        Generate a single image using the Stable Diffusion pipeline
        
        Args:
            prompt: Text prompt for image generation
            negative_prompt: Text prompt for what to avoid in the image
            num_inference_steps: Number of denoising steps
            guidance_scale: How closely to follow the prompt
            height: Image height (multiple of 8)
            width: Image width (multiple of 8)
            
        Returns:
            PIL Image object
        """
        # Get default negative prompt if none provided
        if negative_prompt is None:
            negative_prompt = get_negative_prompt()
        
        # Enhance prompt with realism triggers
        enhanced_prompt = f"realistic, detailed, high-quality, photorealistic, 8k, {prompt}"
        
        # Truncate to avoid token length issues
        enhanced_prompt = self.truncate_prompt(enhanced_prompt)
        negative_prompt = self.truncate_prompt(negative_prompt)
        
        # Make sure height and width are multiples of 8 for VAE
        height = (height // 8) * 8
        width = (width // 8) * 8
        
        print(f"Generating image with prompt: {prompt}")
        
        try:
            # Use the memory-optimized generation
            return self.pipeline(
                prompt=enhanced_prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                height=height,
                width=width
            ).images[0]
        except torch.cuda.OutOfMemoryError:
            print("Out of VRAM! Trying with reduced settings...")
            # Fallback with even more reduced settings
            return self.pipeline(
                prompt=enhanced_prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=20,  # Further reduced steps
                guidance_scale=7.0,
                height=512,  # Minimum size
                width=512    # Minimum size
            ).images[0]
    
    def get_images(self, prompts: List[str], **kwargs) -> List[str]:
        generated_files = []
        total_prompts = len(prompts)
        
        for idx, prompt in enumerate(prompts, 1):
            print(f"Generating image {idx}/{total_prompts}")
            try:
                # Generate the image
                image = self.generate_image(
                    prompt=prompt,
                    **kwargs
                )
                
                # Create a unique filename
                filename = self.create_unique_filename(prompt, f"ai_gen_{idx:02d}")
                
                # Save the image
                image.save(filename)
                print(f"✓ Image saved: {filename}")
                
                generated_files.append(filename)
                
                # Clear VRAM between generations
                if self.device == "cuda":
                    torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"✗ Failed to generate image: {str(e)}")
                if self.device == "cuda":
                    torch.cuda.empty_cache()
        
        return generated_files

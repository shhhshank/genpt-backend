import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler, AutoencoderKL
from PIL import Image
from typing import List, Optional
import asyncio
from pathlib import Path
from get_prompts import get_negative_prompt
from pathlib import Path
import os
import datetime
import hashlib

class RealisticImageGenerator:
    def __init__(self, device="cuda", dtype=torch.float16):
        self.device = device
        self.dtype = dtype
        
        print("Initializing Realistic Vision model...")
        
        # Use RealisticVision v5.1 which balances quality and VRAM usage
        model_id = "SG161222/Realistic_Vision_V5.1_noVAE"
        vae_id = "stabilityai/sd-vae-ft-mse"
        
        # Load the VAE separately for memory efficiency
        vae = AutoencoderKL.from_pretrained(vae_id, torch_dtype=dtype)
        
        # Load the pipeline with memory-efficient settings
        self.pipeline = StableDiffusionPipeline.from_pretrained(
            model_id,
            vae=vae,
            torch_dtype=dtype,
            use_safetensors=True,
            # Cache to disk helps with VRAM
            low_cpu_mem_usage=True,
        )
        self.pipeline = self.pipeline.to(device)
        
        # Memory optimizations
        self.pipeline.enable_attention_slicing(slice_size="auto")
        self.pipeline.enable_vae_tiling()
        self.pipeline.enable_model_cpu_offload()  # Moves components to CPU when not in use
        
        # Use an efficient scheduler
        self.pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
            self.pipeline.scheduler.config, 
            algorithm_type="sde-dpmsolver++", 
            use_karras_sigmas=True
        )
        
        # For better prompt handling
        self.max_token_length = 75  # Safe limit for standard SD models
        
        print("Model initialization complete!")

    def truncate_prompt(self, prompt):
        """Truncate prompt to a safe length for the tokenizer."""
        tokens = self.pipeline.tokenizer.tokenize(prompt)
        if len(tokens) > self.max_token_length:
            # Truncate tokens and decode back to text
            truncated_tokens = tokens[:self.max_token_length]
            truncated_prompt = self.pipeline.tokenizer.convert_tokens_to_string(truncated_tokens)
            print(f"Warning: Prompt truncated from {len(tokens)} to {len(truncated_tokens)} tokens")
            return truncated_prompt
        return prompt
    
    def generate_image(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        num_inference_steps: int = 30,  # Reduced for VRAM efficiency
        guidance_scale: float = 7.0,    # Slightly lower for better generation
        height: int = 640,              # Reduced for VRAM efficiency
        width: int = 640                # Reduced for VRAM efficiency
    ) -> Image.Image:
        """
        Asynchronously generates a realistic image from an abstract concept.
        Optimized for 6GB GPU.
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
        
        try:
            # Use the memory-optimized generation
            return  self.pipeline(
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
                    height=512,             # Minimum size
                    width=512              # Minimum size
                ).images[0]

    def process_prompts(
        self,
        prompts: List[str],
        num_inference_steps: int = 30,
        guidance_scale: float = 7.0,
        height: int = 640,
        width: int = 640
    ):
        """
        Processes a list of prompts, generates images, and saves them to disk.
        Uses relative paths for image storage.
        Optimized for 6GB VRAM.
        """
        # Use a relative path instead of an absolute path
        output_dir = "uploads/generated_image"
        
        # Ensure output directory exists
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Log the absolute path for debugging
        abs_path = os.path.abspath(output_dir)
        print(f"Images will be saved to: {abs_path}")

        # Clear VRAM between generations
        torch.cuda.empty_cache()
        
        generated_files = []
        total_prompts = len(prompts)
        
        for idx, prompt in enumerate(prompts, 1):
            print(f"Generating image {idx}/{total_prompts}: {prompt}")
            try:
                # Generate the image
                image = self.generate_image(
                    prompt=prompt,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    height=height,
                    width=width
                )

                # Create a unique filename based on timestamp and a hash of the prompt
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                prompt_hash = hashlib.md5(prompt.encode()).hexdigest()[:8]
                filename = f"{output_dir}/image_{timestamp}_{prompt_hash}_{idx:02d}.png"
                
                # Ensure filename is unique
                base, ext = os.path.splitext(filename)
                counter = 1
                while os.path.exists(filename):
                    filename = f"{base}_{counter}{ext}"
                    counter += 1
                
                # Save the image
                image.save(filename)
                print(f"Image saved successfully at: {filename}")
                print(f"Absolute path: {os.path.abspath(filename)}")
                
                # Add to list of generated files
                generated_files.append(filename)
                
                # Clear VRAM between generations
                torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"Error generating image for prompt '{prompt}': {e}")
                print("Skipping to next prompt")
                torch.cuda.empty_cache()
        
        # Return all generated file paths
        return generated_files if generated_files else None

    # Helper function to display the saved image
    def display_saved_image(self, image_path):
        """Display an image from the given path using matplotlib."""
        try:
            import matplotlib.pyplot as plt
            import matplotlib.image as mpimg
            
            # Check if the file exists
            if not os.path.exists(image_path):
                print(f"Error: Image file not found at {image_path}")
                print(f"Absolute path would be: {os.path.abspath(image_path)}")
                return False
            
            # Read and display the image
            img = mpimg.imread(image_path)
            plt.figure(figsize=(10, 8))
            plt.imshow(img)
            plt.axis('off')
            plt.title(f'Generated Image: {os.path.basename(image_path)}')
            plt.show()
            return True
        
        except Exception as e:
            print(f"Error displaying image: {e}")
            return False

# Example usage:
# image_paths = image_generator.process_prompts(["a beautiful mountain landscape"])
# if image_paths:
#     display_saved_image(image_paths[0])
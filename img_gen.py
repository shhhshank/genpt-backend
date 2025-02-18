import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from PIL import Image
from typing import List, Optional
import asyncio
from pathlib import Path


class RealisticImageGenerator:
    def __init__(self, model_name="runwayml/stable-diffusion-v1-5", device="cuda", dtype=torch.float16):
        self.device = device
        self.dtype = dtype

        # Load the pipeline
        self.pipeline = StableDiffusionPipeline.from_pretrained(model_name, torch_dtype=dtype)
        self.pipeline = self.pipeline.to(device)

        # Enable optimizations
        self.pipeline.enable_attention_slicing()
        self.pipeline.enable_vae_tiling()

        # Use a better scheduler for quality
        self.pipeline.scheduler = DPMSolverMultistepScheduler.from_config(self.pipeline.scheduler.config)

        # Warm-up
        _ = self.pipeline("warm-up prompt", num_inference_steps=1)

    async def generate_image(
        self,
        prompt: str,
        negative_prompt: Optional[str] = "embedding:BadDream, embedding:UnrealisticDream, embedding:FastNegativeV2, embedding:JuggernautNegative-neg, (deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime:1.4), text, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck, embedding:negative_hand-neg",
        num_inference_steps: int = 50,
        guidance_scale: float = 8.0,
        height: int = 768,
        width: int = 768
    ) -> Image.Image:
        """
        Asynchronously generates an image with optional negative prompts.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                height=height,
                width=width
            ).images[0]
        )

    async def process_prompts(
        self,
        prompts: List[str],
        num_inference_steps: int = 50,
        guidance_scale: float = 8.0,
        height: int = 768,
        width: int = 768
    ):
        """
        Processes a list of prompts, generates images, and saves them to disk.

        :param prompts: List of text prompts for image generation.
        :param output_dir: Directory to save the generated images.
        :param negative_prompt: Things to avoid in the generated images.
        :param num_inference_steps: Number of inference steps for image generation.
        :param guidance_scale: Scale for classifier-free guidance.
        :param height: Height of the output images (multiple of 64).
        :param width: Width of the output images (multiple of 64).
        """
        output_dir = "/uploads/generated_image"
        
        # Ensure output directory exists
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        total_prompts = len(prompts)
        for idx, prompt in enumerate(prompts, 1):
            print(f"Generating image {idx}/{total_prompts}: {prompt}")
            try:
                # Generate the image
                image = await self.generate_image(
                    prompt=prompt,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    height=height,
                    width=width
                )

                # Save the image with a structured filename
                filename = f"{output_dir}/image_{idx:02d}.png"
                image.show()

                print(f"Saved image {idx}/{total_prompts} to {filename}")

            except Exception as e:
                print(f"Error generating image for prompt '{prompt}': {e}")

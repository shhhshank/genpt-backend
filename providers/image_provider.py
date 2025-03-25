# image_provider.py
from abc import ABC, abstractmethod
import os
from typing import List, Optional, Dict, Any
import datetime
import hashlib

class ImageProvider(ABC):
    """Abstract base class for different image generation/fetching strategies"""
    
    def __init__(self, save_dir: str = "uploads/generated_image"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
    
    @abstractmethod
    def get_images(self, prompts: List[str], **kwargs) -> List[str]:
        """
        Generate or fetch images based on prompts
        Returns a list of file paths to the saved images
        """
        pass
    
    def create_unique_filename(self, prompt: str, prefix: str, extension: str = '.png') -> str:
        """Generate a unique filename based on timestamp and prompt hash"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        prompt_hash = hashlib.md5(prompt.encode()).hexdigest()[:8]
        filename = f"{self.save_dir}/{prefix}_{timestamp}_{prompt_hash}{extension}"
        
        # Ensure filename is unique
        base, ext = os.path.splitext(filename)
        counter = 1
        while os.path.exists(filename):
            filename = f"{base}_{counter}{ext}"
            counter += 1
            
        return filename

    @classmethod
    def build_image_prompt(cls, slide_content: str, image_provider: str = 'ai', topic: str = None) -> str:
        # For search-based providers (google, stock, web), just return the slide title
        if image_provider in ['google', 'stock', 'web']:
            return slide_content['title']  # Use slide title directly
        
        # For AI provider, keep the existing complex prompt generation
        else:  # Default 'ai' provider
            # AI art generation prompt
            prefix = """You are an expert in AI-generated art prompt engineering, specializing in designing visuals 
            for PowerPoint presentations. You strictly follow the provided comprehensive guide on crafting the most 
            effective AI art prompts—this guide is your absolute reference. Your task is to generate a precise, structured 
            AI art prompt based on the given slide content. 

            Key Rules:
            - **Strictly adhere to the guide** for formatting, style, and content structure.
            - **Only generate one AI art prompt**—no explanations, variations, or extra text.
            - **Optimize the prompt for Stable Diffusion** to ensure high-quality, relevant images.
            - **The prompt should be under 70 words
            """

            # Retrieve the AI art guide
            from get_prompts import get_image_guide_prompt
            guide = get_image_guide_prompt()
            
            guide_section = f"\n\n-- Start of Comprehensive Guide --\n{guide}\n-- End of Comprehensive Guide --\n"

            output_guide = """
            -- Start of Prompt Output Guide -- 
            - **Output must contain only the final AI art prompt.**
            - **The format must strictly follow this output structure:**  
              <<prompt>> example:  
              <<image prompt which will be used to query the image>>  
            -- End of Prompt Output Guide -- 
            """
            
            # For AI provider, return early with the full guide included
            slide_section = f"\n-- Start of Slide Content --\n{slide_content}\n-- End of Slide Content --\n"
            
            if image_provider == 'web':  
                final_prompt = prefix + slide_section + output_guide
            else:
                final_prompt = prefix + guide_section + slide_section + output_guide
                
            return final_prompt

        # For web and local providers, build a simpler prompt
        slide_section = f"\n-- Slide Content --\n{slide_content}\n-- End of Slide Content --\n"
        final_prompt = prefix + slide_section + output_guide
        
        return final_prompt
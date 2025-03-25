import os
import random
import shutil
from typing import List, Optional
from .image_provider import ImageProvider

class LocalImageProvider(ImageProvider):
    """Provides images from a local directory based on categorization"""
    
    def __init__(self, 
                 save_dir: str = "uploads/generated_image",
                 image_library_path: str = "image_library"):
        """
        Initialize the Local Image Provider
        
        Args:
            save_dir: Directory to save copied images
            image_library_path: Path to local image library organized by categories
        """
        super().__init__(save_dir)
        self.image_library_path = image_library_path
        self._validate_library()
        
    def _validate_library(self):
        """Validate that the image library exists and contains images"""
        if not os.path.exists(self.image_library_path):
            print(f"Image library path does not exist: {self.image_library_path}")
            os.makedirs(self.image_library_path, exist_ok=True)
            
        # Create some default categories if empty
        if not os.listdir(self.image_library_path):
            print("Empty image library. Creating default categories.")
            default_categories = ["business", "nature", "technology", "people", "abstract"]
            for category in default_categories:
                os.makedirs(os.path.join(self.image_library_path, category), exist_ok=True)
    
    def get_images(self, prompts: List[str], **kwargs) -> List[str]:
        selected_files = []
        valid_categories = self._get_available_categories()
        
        for idx, prompt in enumerate(prompts, 1):
            print(f"Finding local image {idx}/{len(prompts)}: '{prompt}'")
            
            category = self._select_category_for_prompt(prompt, valid_categories)
            image_path = self._get_image_from_category(category)
            
            if image_path:
                extension = os.path.splitext(image_path)[1]
                new_filename = self.create_unique_filename(prompt, f"local_{category}", extension)
                
                try:
                    shutil.copy2(image_path, new_filename)
                    print(f"✓ Image copied to {new_filename}")
                    selected_files.append(new_filename)
                except Exception as e:
                    print(f"✗ Error copying image: {str(e)}")
            else:
                print(f"✗ No suitable image found for: '{prompt}'")
        
        return selected_files
    
    def _get_available_categories(self) -> List[str]:
        """Get available categories in the image library"""
        if not os.path.exists(self.image_library_path):
            return []
            
        return [d for d in os.listdir(self.image_library_path) 
                if os.path.isdir(os.path.join(self.image_library_path, d))]
    
    def _derive_categories_from_prompts(self, prompts: List[str], available_categories: List[str]) -> List[str]:
        """Derive likely categories from prompts"""
        # Simple keyword matching for now
        derived_categories = set()
        
        for prompt in prompts:
            prompt_lower = prompt.lower()
            for category in available_categories:
                if category.lower() in prompt_lower:
                    derived_categories.add(category)
        
        # If no matches, use all available categories
        if not derived_categories:
            derived_categories = set(available_categories)
            
        return list(derived_categories)
    
    def _select_category_for_prompt(self, prompt: str, categories: List[str]) -> str:
        """Select the most appropriate category for a prompt"""
        # For now, just check if any category name is in the prompt
        prompt_lower = prompt.lower()
        
        for category in categories:
            if category.lower() in prompt_lower:
                return category
                
        # If no match, return a random category
        return random.choice(categories)
    
    def _get_image_from_category(self, category: str) -> Optional[str]:
        """Get a random image from a specific category"""
        category_path = os.path.join(self.image_library_path, category)
        
        if not os.path.exists(category_path):
            print(f"Category path does not exist: {category_path}")
            return None
            
        # Get all files in the category directory
        image_files = []
        for file in os.listdir(category_path):
            file_path = os.path.join(category_path, file)
            if os.path.isfile(file_path) and self._is_image_file(file):
                image_files.append(file_path)
        
        if not image_files:
            print(f"No image files found in category: {category}")
            return None
            
        # Return a random image from the category
        return random.choice(image_files)
    
    def _is_image_file(self, filename: str) -> bool:
        """Check if a file is an image based on extension"""
        image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp']
        return any(filename.lower().endswith(ext) for ext in image_extensions)

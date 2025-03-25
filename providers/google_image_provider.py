from typing import List
import os
from googleapiclient.discovery import build
from .image_provider import ImageProvider
import requests
import time

class GoogleImageProvider(ImageProvider):
    """Provider for Google Custom Search Images"""
    
    def __init__(self, api_key: str, cx_id: str, save_dir: str = "uploads/generated_image"):
        super().__init__(save_dir)
        self.api_key = api_key
        self.cx_id = cx_id
        self.service = build('customsearch', 'v1', developerKey=self.api_key)
        
        # Configurable search parameters
        self.search_config = {
            'default_sites': ['shutterstock.com', 'gettyimages.com', 'flickr.com'],
            'excluded_sites': ['pinterest.*', 'instagram.*'],
            'min_image_size': '2MP',
            'preferred_types': ['jpg', 'png']
        }

    def build_search_query(self, prompt: str, **kwargs) -> str:
        """Add Google-specific search operators to the basic search terms"""
        search_terms = prompt.strip()
        return f"{search_terms}"

    def get_images(self, prompts: List[str]) -> List[str]:
        results = []
        for prompt in prompts:
            # Try different search variations until we get an image
            search_variations = [
                prompt,  # Original title
                f"{prompt} photo",  # Add "photo" keywordpip install google

                f"{prompt} image",  # Add "image" keyword
                f"{prompt} illustration",  # Try illustration
                "generic " + prompt  # Add "generic" prefix as last resort
            ]
            
            for search_query in search_variations:
                try:
                    search_result = self.service.cse().list(
                        q=self.build_search_query(search_query),
                        cx=self.cx_id,
                        searchType='image',
                        num=1,
                        imgType='photo',
                        safe='active',
                        imgSize='LARGE',
                        fileType='jpg|png'
                    ).execute()

                    if 'items' in search_result:
                        image_url = search_result['items'][0]['link']
                        timestamp = int(time.time() * 1000)
                        filename = f"{self.save_dir}/google_{timestamp}.jpg"
                        
                        response = requests.get(image_url)
                        if response.status_code == 200:
                            os.makedirs(os.path.dirname(filename), exist_ok=True)
                            with open(filename, 'wb') as f:
                                f.write(response.content)
                            results.append(filename)
                            print(f"✓ Found image using query: {search_query}")
                            break  # Exit the variations loop once we have an image
                        
                except Exception as e:
                    print(f"✗ Search failed for '{search_query}': {str(e)}")
                    continue
            
            if not results:  # If no image found after all variations
                print(f"✗ No image found for any variation of: {prompt}")
                results.append(None)
                
        return results 
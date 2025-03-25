from typing import List, Optional, Dict
from PIL import Image
from io import BytesIO
from .image_provider import ImageProvider
from pexels_api import API as PexelsAPI
import requests
import time
import os

class StockImageProvider(ImageProvider):
    """Provider for Stock Image APIs (Pexels, Unsplash, Pixabay)"""
    
    def __init__(self, api_keys: Dict[str, str] = None, save_dir: str = "uploads/generated_image"):
        super().__init__(save_dir)
        self.api_keys = api_keys or {
            "unsplash": "wxEgsxxLRwKPNdXdrLxcQ9zWZ6v_WbxGS4iSHbMExJ4",
            "pexels": "nZbpToi4321FsmtBjb3Ddt9Lo64VRzoV32bzIDT25yuqtbWKo4GJUNrl",
            "pixabay": "49466255-49e89b2abd761baf6e5970275"
        }
        self.pexels = PexelsAPI(self.api_keys["pexels"])

    def get_images(self, prompts: List[str], provider: str = "pexels", fallback_providers: List[str] = None) -> List[str]:
        """
        Fetch images from stock providers with fallback support
        
        Args:
            prompts: List of search queries
            provider: Primary provider to use ("pexels", "unsplash", "pixabay")
            fallback_providers: List of providers to try if primary fails
        """
        fallback_providers = fallback_providers or ["pixabay", "unsplash"]
        downloaded_files = []

        for prompt in prompts:
            image_path = self._fetch_from_provider(prompt, provider)

            if not image_path:
                for fallback in fallback_providers:
                    image_path = self._fetch_from_provider(prompt, fallback)
                    if image_path:
                        break

            if image_path:
                downloaded_files.append(image_path)
            else:
                print(f"✗ No image found for: {prompt}")

        return downloaded_files

    def _fetch_from_provider(self, query: str, provider: str) -> Optional[str]:
        try:
            fetch_methods = {
                "unsplash": self._fetch_from_unsplash,
                "pexels": self._fetch_from_pexels,
                "pixabay": self._fetch_from_pixabay,
            }
            fetch_method = fetch_methods.get(provider)
            if fetch_method:
                return fetch_method(query)
            else:
                print(f"Unknown provider: {provider}")
                return None
        except Exception as e:
            print(f"Error fetching from {provider}: {e}")
            return None

    def _fetch_from_pixabay(self, query: str) -> Optional[str]:
        api_key = self.api_keys.get("pixabay")
        if not api_key:
            print("No API key provided for Pixabay")
            return None

        url = f"https://pixabay.com/api/?key={api_key}&q={query}&image_type=photo&per_page=1&min_width=640&min_height=640"
        
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            
            if data.get('hits'):
                image_url = data['hits'][0]['largeImageURL']
                return self._process_image(requests.get(image_url), query, "pixabay")
            return None
        except Exception as e:
            print(f"Error fetching from Pixabay: {e}")
            return None

    def _fetch_from_pexels(self, query: str) -> Optional[str]:
        try:
            self.pexels.search(query, results_per_page=1)
            photos = self.pexels.get_entries()
            if photos:
                image_response = requests.get(photos[0].original)
                return self._process_image(image_response, query, "pexels")
            return None
        except Exception as e:
            print(f"Error fetching from Pexels: {e}")
            return None

    def _fetch_from_unsplash(self, query: str) -> Optional[str]:
        api_key = self.api_keys.get("unsplash")
        if not api_key:
            print("No API key provided for Unsplash")
            return None

        headers = {'Authorization': f'Client-ID {api_key}'}
        url = "https://api.unsplash.com/photos/random"
        params = {
            'query': query,
            'orientation': 'landscape',
            'w': 640,
            'h': 640
        }

        try:
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            data = response.json()
            image_url = data['urls']['regular']
            image_response = requests.get(image_url)
            return self._process_image(image_response, query, "unsplash")
        except Exception as e:
            print(f"Error fetching from Unsplash: {e}")
            return None

    def _process_image(self, response: requests.Response, query: str, source: str) -> Optional[str]:
        try:
            image = Image.open(BytesIO(response.content))
            extension = ".jpg"
            filename = self.create_unique_filename(query, f"{source}", extension)
            image.save(filename, "JPEG")
            print(f"Successfully saved image to {filename}")
            return filename
        except Exception as e:
            print(f"Error processing image: {e}")
            return None 
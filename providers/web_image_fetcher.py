import requests
from typing import List, Optional, Dict
from PIL import Image
from io import BytesIO
from providers.image_provider import ImageProvider
from pexels_api import API as PexelsAPI
import os
import json

class WebImageFetcher(ImageProvider):
    """Fetches images using SDKs for Pexels and Pixabay, and requests for Unsplash."""

    def __init__(self, save_dir: str = "uploads/generated_image", api_keys: Dict[str, str] = None):
        super().__init__(save_dir)
        self.api_keys = api_keys or {
            "unsplash":"wxEgsxxLRwKPNdXdrLxcQ9zWZ6v_WbxGS4iSHbMExJ4",
            "pexels":"nZbpToi4321FsmtBjb3Ddt9Lo64VRzoV32bzIDT25yuqtbWKo4GJUNrl",
            "pixabay": "49466255-49e89b2abd761baf6e5970275"
        }
        self.pexels = PexelsAPI(self.api_keys["pexels"])

    def get_images(self, prompts: List[str], provider: str = "pexels", fallback_providers: List[str] = None) -> List[str]:
        fallback_providers = fallback_providers or ["pixabay", "unsplash"]
        downloaded_files = []

        for prompt in prompts:
            image_path = self._fetch_from_provider(prompt, provider)

            if not image_path and fallback_providers:
                for fallback in fallback_providers:
                    image_path = self._fetch_from_provider(prompt, fallback)
                    if image_path:
                        break

            if image_path:
                downloaded_files.append(image_path)

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
                print(f"✗ Unknown provider: {provider}")
                return None
        except Exception as e:
            print(f"✗ Error fetching from {provider}: {str(e)}")
            return None

    def _fetch_from_unsplash(self, query: str) -> Optional[str]:
        api_key = self.api_keys.get("unsplash")
        if not api_key:
            print("✗ No API key provided for Unsplash")
            return None

        url = f"https://api.unsplash.com/photos/random?query={query}&client_id={api_key}&w=640&h=640&fit=crop"

        try:
            response = requests.get(url)
            response.raise_for_status()
            json_data = response.json()
            image_url = json_data["urls"]["regular"]
            image_response = requests.get(image_url, stream=True)
            image_response.raise_for_status()
            return self._process_image(image_response, query, "unsplash")
        except requests.exceptions.RequestException as e:
            print(f"✗ Error fetching from Unsplash: {str(e)}")
            return None
        except (json.JSONDecodeError, KeyError) as e:
            print(f"✗ Error parsing Unsplash response: {str(e)}")
            return None

    def _fetch_from_pexels(self, query: str) -> Optional[str]:
        try:
            self.pexels.search(query, results_per_page=1)
            photos = self.pexels.get_entries()
            if photos:
                return self._fetch_image(photos[0].original, query, "pexels")
            return None
        except Exception as e:
            print(f"✗ Error fetching from Pexels: {str(e)}")
            return None

    def _fetch_from_pixabay(self, query: str) -> Optional[str]:
        api_key = self.api_keys.get("pixabay")
        if not api_key:
            print("✗ No API key provided for Pixabay")
            return None
        url = f"https://pixabay.com/api/?key={api_key}&q={query}&image_type=photo&per_page=1&min_width=640&min_height=640"
        return self._fetch_image(url, query, "pixabay")

    def _fetch_image(self, url: str, query: str, source: str) -> Optional[str]:
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            return self._process_image(response, query, source)
        except requests.exceptions.RequestException as e:
            print(f"✗ Error downloading image: {str(e)}")
            return None

    def _process_image(self, response: requests.Response, query: str, source: str) -> Optional[str]:
        try:
            image = Image.open(BytesIO(response.content))
            image = self._resize_to_square(image, 640)
            extension = ".jpg"
            filename = self.create_unique_filename(query, f"{source}", extension)
            image.save(filename, "JPEG")
            print(f"✓ Image saved to {filename}")
            return filename
        except Exception as e:
            print(f"✗ Error processing image: {str(e)}")
            return None

    def _resize_to_square(self, image: Image.Image, size: int) -> Image.Image:
        min_side = min(image.size)
        if image.width > image.height:
            left = (image.width - min_side) / 2
            image = image.crop((left, 0, left + min_side, min_side))
        elif image.height > image.width:
            top = (image.height - min_side) / 2
            image = image.crop((0, top, min_side, top + min_side))
        return image.resize((size, size), Image.LANCZOS)
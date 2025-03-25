from typing import Dict, Optional, Any
import os
from .image_provider import ImageProvider
from .google_image_provider import GoogleImageProvider
from .stock_image_provider import StockImageProvider
from .ai_image_generator import AIImageGenerator

class ImageFactory:
    """Factory class to create image providers"""
    
    @staticmethod
    def create_provider(provider_type: str = 'google', config: Dict[str, Any] = None) -> Optional[ImageProvider]:
        """
        Create and return an image provider instance
        
        Args:
            provider_type: Type of provider ('ai', 'google', 'stock')
            config: Configuration options for the provider
        """
        config = config or {}
        try:
            if provider_type == 'ai':
                return AIImageGenerator(**config)
            elif provider_type == 'google':
                api_key = "AIzaSyDdxB2ZCOoMB5ciwpSPBP3olsvLmGp8Wy0"
                cx_id = "3539b4c3899364950"
                return GoogleImageProvider(api_key=api_key, cx_id=cx_id)
            elif provider_type == 'stock':
                return StockImageProvider()
            else:
                print(f"✗ Unknown provider type: {provider_type}")
                return None
        except Exception as e:
            print(f"✗ Provider error: {str(e)}")
            return None


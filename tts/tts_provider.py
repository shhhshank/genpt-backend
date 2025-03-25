from abc import ABC, abstractmethod
import torch
import os
from typing import Optional, Dict, Any
import requests
from TTS.api import TTS
from fairseq.checkpoint_utils import load_model_ensemble_and_task
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
import numpy as np
import soundfile as sf
from gtts import gTTS

class TTSProvider(ABC):
    @abstractmethod
    def generate_speech(self, text: str, output_path: str) -> bool:
        pass

class CoquiTTSProvider(TTSProvider):
    def __init__(self, model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tts = TTS(model_name).to(self.device)
        
    def generate_speech(self, text: str, output_path: str) -> bool:
        try:
            self.tts.tts_to_file(
                text=text,
                file_path=output_path,
                speaker_wav="path/to/reference_audio.wav",  # Optional: for voice cloning
                language="en"
            )
            return True
        except Exception as e:
            print(f"Coqui TTS error: {str(e)}")
            return False

class FacebookMMSProvider(TTSProvider):
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        checkpoint = torch.hub.load_state_dict_from_url(
            "https://dl.fbaipublicfiles.com/mms/tts/mms_tts_en.pt",
            map_location=self.device
        )
        self.model, self.cfg, self.task = load_model_ensemble_and_task([checkpoint])
        self.model = self.model[0].to(self.device)
        
    def generate_speech(self, text: str, output_path: str) -> bool:
        try:
            with torch.no_grad():
                sample = self.task.get_batch_from_text([text])
                wav, sr = self.model.generate_speech(sample)
                sf.write(output_path, wav.cpu().numpy(), sr)
            return True
        except Exception as e:
            print(f"Facebook MMS error: {str(e)}")
            return False

class MicrosoftSpeechT5Provider(TTSProvider):
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
        self.model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts").to(self.device)
        self.vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(self.device)
        
    def generate_speech(self, text: str, output_path: str) -> bool:
        try:
            inputs = self.processor(text=text, return_tensors="pt").to(self.device)
            speech = self.model.generate_speech(inputs["input_ids"], self.vocoder)
            sf.write(output_path, speech.cpu().numpy(), 16000)
            return True
        except Exception as e:
            print(f"Microsoft SpeechT5 error: {str(e)}")
            return False

class TTSFactory:
    @staticmethod
    def create_provider(provider_type: str, **kwargs) -> TTSProvider:
        providers = {
            "coqui": CoquiTTSProvider,
            "facebook": FacebookMMSProvider,
            "microsoft": MicrosoftSpeechT5Provider,
            "gtts": lambda: GTTSProvider()
        }
        
        if provider_type not in providers:
            raise ValueError(f"Unknown provider type: {provider_type}")
            
        return providers[provider_type](**kwargs) 
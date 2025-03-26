import os
import re
import subprocess
import shutil
from typing import List, Dict, Optional
import google.generativeai as genai
from gtts import gTTS
import time
from PIL import Image
import numpy as np
import requests
from flask import Blueprint, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from google.cloud import texttospeech

class ImagesVideoGenerator:
    def __init__(self, output_dir: str = "uploads/generated_videos", min_slide_duration: int = 15, max_slide_duration: int = 30):
        self.output_dir = output_dir
        self.temp_dir = os.path.join("uploads", "temp_video")
        self.audio_dir = os.path.join(self.temp_dir, "audio")
        self.image_dir = os.path.join(self.temp_dir, "images")
        self.min_slide_duration = min_slide_duration
        self.max_slide_duration = max_slide_duration
        self.words_per_minute = 150  # Average speaking rate
        
        # ElevenLabs API configuration
        self.elevenlabs_keys = [
            "sk_5ca25de453c020da35550ca308a292f90fb45594ad2193a6",
            "sk_47a9ada9e4339f29bcb1bd6a28c99383d44bf9005cd514f0",
            "sk_d8e5f67ca80184554c8165e04821fd81d8575ba4575dbca3",
            "sk_105c1bea704b11f9fee21f5735cba471147cf13714b4c8bc",
            "sk_4fb2de074dd4b122bf1e3d5c510f4ea7f33ab5c6fcc600f0"
        ]
        self.voice_id = "JBFqnCBsd6RMkjVDRZzb"  # Antoni voice ID
        
        self.setup_directories()
        
        # Initialize Gemini API
        API_KEY = "AIzaSyB1U3NavO8CvkQ2pk0NFpEf_NKYJPCnAPk"
        print("Configuring Gemini API")
        genai.configure(api_key=API_KEY)
        
        # Configure the model
        generation_config = {
            "temperature": 0.4,
            "top_p": 1,
            "top_k": 32,
            "max_output_tokens": 2048,
        }
        
        # Create a Gemini model instance
        print("Creating Gemini model instance with gemini-1.5-pro")
        self.model = genai.GenerativeModel(
            model_name="gemini-1.5-pro",
            generation_config=generation_config
        )

        # Configure Google Cloud TTS
        self.tts_client = texttospeech.TextToSpeechClient()
        self.voice = texttospeech.VoiceSelectionParams(
            language_code="en-IN",  # Indian English
            name="en-IN-Neural2-A",  # Female Indian English voice
            ssml_gender=texttospeech.SsmlVoiceGender.FEMALE
        )
        self.audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3,
            speaking_rate=1.0,
            pitch=0.0,
            sample_rate_hertz=44100
        )

    def setup_directories(self):
        """Ensure all required directories exist"""
        for dir_path in [self.output_dir, self.temp_dir, self.audio_dir, self.image_dir]:
            os.makedirs(dir_path, exist_ok=True)

    def process_images(self, image_paths: List[str]) -> List[Dict]:
        """Process a list of images and extract information"""
        slides_content = []
        
        for idx, image_path in enumerate(image_paths):
            print(f"Processing image {idx + 1}/{len(image_paths)}: {image_path}")
            try:
                # Verify image exists
                if not os.path.exists(image_path):
                    print(f"✗ Image not found: {image_path}")
                    continue
                
                # Load and prepare the image for analysis
                with Image.open(image_path) as img:
                    # Convert image to RGB (removing alpha channel)
                    if img.mode in ('RGBA', 'LA', 'P'):
                        # Create white background
                        background = Image.new('RGB', img.size, (255, 255, 255))
                        if img.mode == 'P':
                            img = img.convert('RGBA')
                        # Paste using alpha channel as mask
                        background.paste(img, mask=img.split()[3] if img.mode == 'RGBA' else None)
                        img = background
                    elif img.mode != 'RGB':
                        img = img.convert('RGB')
                    
                    # Resize image if too large for API
                    if max(img.size) > 3000:
                        img.thumbnail((3000, 3000), Image.LANCZOS)
                    
                    # Save a copy to our working directory
                    processed_path = os.path.join(self.image_dir, f"image_{idx}.jpg")
                    img.save(processed_path, "JPEG", quality=95)
                    
                    print(f"✓ Saved processed image: {processed_path} ({img.size[0]}x{img.size[1]})")
                
                # Extract content using Gemini
                image_content = self._analyze_image(processed_path)
                
                slides_content.append({
                    'index': idx,
                    'image_path': processed_path,
                    'title': image_content['title'],
                    'points': image_content['description'],
                    'context': image_content['context']
                })
                
                print(f"✓ Processed image {idx + 1}")
                
            except Exception as e:
                print(f"✗ Error processing image {idx + 1}: {str(e)}")
                try:
                    # Fallback method for problematic images
                    with Image.open(image_path) as img:
                        # Force conversion to RGB with white background
                        background = Image.new('RGB', img.size, (255, 255, 255))
                        if img.mode == 'P':
                            img = img.convert('RGBA')
                        if img.mode == 'RGBA':
                            background.paste(img, mask=img.split()[3])
                        else:
                            background.paste(img)
                        
                        processed_path = os.path.join(self.image_dir, f"image_{idx}.jpg")
                        background.save(processed_path, "JPEG", quality=95)
                        
                        print(f"✓ Saved image using fallback method: {processed_path}")
                        
                        image_content = self._analyze_image(processed_path)
                        slides_content.append({
                            'index': idx,
                            'image_path': processed_path,
                            'title': image_content['title'],
                            'points': image_content['description'],
                            'context': image_content['context']
                        })
                except Exception as e2:
                    print(f"✗ Fallback method also failed: {str(e2)}")
                    # Add minimal information for failed images
                    slides_content.append({
                        'index': idx,
                        'image_path': image_path,
                        'title': f"Image {idx + 1}",
                        'points': ["No description available"],
                        'context': "Unable to analyze image"
                    })
        
        if not slides_content:
            raise ValueError("No valid images were processed")
        
        return slides_content

    def _analyze_image(self, image_path: str) -> Dict:
        """Analyze image content using Gemini"""
        try:
            # Read image as binary
            with open(image_path, "rb") as img_file:
                image_bytes = img_file.read()
            
            # Create Gemini image parts
            image_parts = [{"mime_type": "image/jpeg", "data": image_bytes}]
            
            # Prompt for image analysis
            prompt = """
            Analyze this image in detail and provide the following information:
            
            1. A concise title (5-7 words) that captures the main subject or theme
            2. 3-5 key points about the image content (each 10-15 words)
            3. Brief context about what's shown (30-50 words)
            
            Format your response EXACTLY as follows:
            <<
            {
              "title": "Main Subject or Theme of Image",
              "description": [
                "First key point about the image",
                "Second key point about the image",
                "Third key point about the image"
              ],
              "context": "Brief contextual information about what's shown in the image"
            }
            >>
            """
            
            # Call Gemini API with image
            response = self.model.generate_content([prompt] + image_parts)
            result = self._extract_json(response.text)
            
            # Validate and return the result
            if not result or not all(k in result for k in ["title", "description", "context"]):
                raise ValueError("Invalid response format from image analysis")
            
            return result
            
        except Exception as e:
            print(f"✗ Error analyzing image with Gemini: {str(e)}")
            # Return fallback analysis
            return {
                "title": os.path.basename(image_path),
                "description": ["Image content could not be analyzed"],
                "context": "No context available"
            }

    def _extract_json(self, text: str) -> Dict:
        """Extract JSON from the Gemini response"""
        try:
            # Extract text between << and >> markers
            pattern = r"<<\s*(\{.*?\})\s*>>"
            match = re.search(pattern, text, re.DOTALL)
            
            if match:
                json_str = match.group(1)
                # Handle potential single quotes or other issues
                import json
                result = json.loads(json_str)
                return result
            
            # Fallback: try to find any JSON object in the text
            import json
            pattern = r"\{[^{]*\"title\"[^}]*\}"
            match = re.search(pattern, text, re.DOTALL)
            if match:
                return json.loads(match.group(0))
                
            raise ValueError("Could not extract JSON from response")
            
        except Exception as e:
            print(f"✗ Error extracting JSON: {str(e)}")
            print(f"Raw response: {text}")
            return {}

    def generate_script(self, slides_content: List[Dict]) -> List[Dict]:
        """Generate narration scripts for images"""
        scripts = []
        
        for slide in slides_content:
            # Calculate target word count based on duration limits
            min_words = 15
            max_words = 9999999999
            
            prompt = f"""
            You are a professional narrator describing images for a video. Create an engaging script.
            
            Image title: {slide['title']}
            Key points: {slide['points']}
            Context: {slide['context']}
            
            STRICT Requirements:
            1. Word count must be between {min_words} and {max_words} words
            2. Use natural, conversational language as if narrating a documentary
            3. Include brief pauses (marked with "...")
            4. Use *asterisks* for emphasis on key terms
            5. Start with an engaging introduction
            6. Describe what's visible in the image
            7. Add relevant context or explanation
            8. End with a thoughtful conclusion
            
            Format output as:
            <<A man standing in a field>>
            """
            
            try:
                response = self.model.generate_content(prompt)
                script = self._extract_prompt(response.text)
                
                # Verify word count
                word_count = len(script.split())
                if word_count < min_words or word_count > max_words:
                    print(f"⚠️ Script word count ({word_count}) outside target range ({min_words}-{max_words})")
                    
                    adjust_prompt = f"""
                    Rewrite this script to contain {min_words}-{max_words} words while maintaining all key points:
                    
                    {script}
                    
                    Keep the same style and format, but adjust length.
                    Format: <<A mam in a field>>
                    """
                    
                    response = self.model.generate_content(adjust_prompt)
                    script = self._extract_prompt(response.text)
                
                scripts.append({
                    'slide_index': slide['index'],
                    'script': script,
                    'target_duration': (self.min_slide_duration + self.max_slide_duration) / 2
                })
                
                print(f"✓ Generated script for image {slide['index'] + 1} "
                      f"({len(script.split())} words)")
                
            except Exception as e:
                print(f"✗ Error generating script for image {slide['index'] + 1}: {str(e)}")
                # Provide a basic fallback script
                fallback_script = self._generate_fallback_script(slide)
                scripts.append({
                    'slide_index': slide['index'],
                    'script': fallback_script,
                    'target_duration': self.min_slide_duration
                })
        
        return scripts

    def _generate_fallback_script(self, slide: Dict) -> str:
        """Generate a simple fallback script if main generation fails"""
        if isinstance(slide['points'], list):
            points_text = '. '.join(slide['points'])
        else:
            points_text = slide['points']
            
        return f"""In this image, we can see {slide['title']}... {points_text}... 
        This image shows important visual information related to the topic."""

    def _extract_prompt(self, text: str) -> str:
        """Extract text between << and >> markers with error handling"""
        pattern = r"<<(.+?)>>"
        match = re.search(pattern, text, re.DOTALL)
        if not match:
            # If no markers found, try to extract any reasonable text
            lines = text.split('\n')
            content_lines = [line.strip() for line in lines if line.strip() and not line.startswith('-')]
            return ' '.join(content_lines)
        return match.group(1).strip()

    def _try_elevenlabs_api(self, script: Dict, audio_path: str, api_key: str, voice_id: str) -> bool:
        """Try generating audio with ElevenLabs using specified API key"""
        try:
            url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
            
            headers = {
                "xi-api-key": api_key,
                "Content-Type": "application/json"
            }
            
            data = {
                "text": script['script'],
                "model_id": "eleven_multilingual_v2"
            }
            
            response = requests.post(
                f"{url}?output_format=mp3_44100_128",
                headers=headers,
                json=data
            )
            
            if response.status_code == 200:
                with open(audio_path, 'wb') as f:
                    f.write(response.content)
                return True
            elif response.status_code in [401, 403]:
                print(f"⚠️ API key expired or invalid: {response.text}")
                return False
            else:
                print(f"⚠️ ElevenLabs API error: {response.text}")
                return False
                
        except Exception as e:
            print(f"⚠️ Error with API key: {str(e)}")
            return False

    def generate_audio(self, scripts: List[Dict]) -> List[str]:
        """Generate audio using Google Cloud TTS with gTTS as fallback"""
        audio_paths = []
        
        for script in scripts:
            audio_path = os.path.join(self.audio_dir, f"slide_{script['slide_index']}.mp3")
            
            try:
                # Prepare the synthesis input
                synthesis_input = texttospeech.SynthesisInput(
                    text=script['script']
                )
                
                print(f"Generating audio for slide {script['slide_index'] + 1} using Google Cloud TTS")
                
                # Perform the text-to-speech request
                response = self.tts_client.synthesize_speech(
                    input=synthesis_input,
                    voice=self.voice,
                    audio_config=self.audio_config
                )
                
                # Write the response to the output file
                with open(audio_path, "wb") as out:
                    out.write(response.audio_content)
                    
                print(f"✓ Generated audio with Google Cloud TTS for slide {script['slide_index'] + 1}")
                
            except Exception as e:
                print(f"✗ Error with Google Cloud TTS: {str(e)}")
                print("Falling back to gTTS...")
                try:
                    audio_path = self._generate_single_audio_gtts(script)
                except Exception as e2:
                    print(f"✗ Error in gTTS fallback: {str(e2)}")
                    print("Creating silent audio as last resort")
                    audio_path = self._generate_silent_audio(self.min_slide_duration)
            
            # Verify audio duration meets requirements
            audio_path = self._verify_audio_duration(audio_path, script['target_duration'])
            
            audio_paths.append(audio_path)
            time.sleep(0.1)  # Small delay between requests
        
        return audio_paths

    def _generate_single_audio_gtts(self, script: Dict) -> str:
        """Generate a single audio file using gTTS (fallback method)"""
        try:
            audio_path = os.path.join(self.audio_dir, f"slide_{script['slide_index']}.mp3")
            
            # Add pauses for better pacing
            text_with_pauses = self._add_speech_pauses(script['script'])
            
            # Generate audio with gTTS
            tts = gTTS(text=text_with_pauses, lang='en-in', slow=False)  # Using Indian English
            tts.save(audio_path)
            
            print(f"✓ Generated audio with gTTS for slide {script['slide_index'] + 1}")
            return audio_path
            
        except Exception as e:
            print(f"✗ Error generating gTTS audio: {str(e)}")
            raise

    def _add_speech_pauses(self, text: str) -> str:
        """Add natural pauses to the text for better gTTS output"""
        # Add pause after sentences
        text = text.replace('. ', '... ')
        
        # Add pause after commas
        text = text.replace(', ', '... ')
        
        # Add pause for emphasis
        text = text.replace('*', '... ')
        
        return text

    def _generate_silent_audio(self, duration: float) -> str:
        """Generate silent audio file as last resort"""
        try:
            audio_path = os.path.join(self.audio_dir, f"silence_{int(time.time())}.mp3")
            
            # Generate silent audio using FFmpeg
            subprocess.run([
                'ffmpeg', '-y',
                '-f', 'lavfi',
                '-i', f'anullsrc=r=44100:cl=stereo:d={duration}',
                '-c:a', 'libmp3lame',
                '-q:a', '2',
                audio_path
            ], check=True)
            
            return audio_path
        except Exception as e:
            print(f"✗ Error generating silent audio: {str(e)}")
            raise

    def _verify_audio_duration(self, audio_path: str, target_duration: float) -> str:
        """Verify and adjust audio duration to meet target"""
        actual_duration = self._get_audio_duration(audio_path)
        
        if actual_duration < self.min_slide_duration:
            # Add silence to meet minimum duration
            print(f"⚠️ Audio too short ({actual_duration:.1f}s), extending to {self.min_slide_duration}s")
            silence_duration = self.min_slide_duration - actual_duration
            silence_path = self._generate_silent_audio(silence_duration)
            
            # Combine original audio with silence
            combined_path = audio_path + '.adjusted.mp3'
            subprocess.run([
                'ffmpeg', '-y',
                '-i', 'concat:' + audio_path + '|' + silence_path,
                '-acodec', 'copy',
                combined_path
            ], check=True)
            
            os.remove(silence_path)
            os.replace(combined_path, audio_path)
            
        elif actual_duration > self.max_slide_duration:
            # Trim audio to maximum duration
            print(f"⚠️ Audio too long ({actual_duration:.1f}s), trimming to {self.max_slide_duration}s")
            temp_path = audio_path + '.temp.mp3'
            subprocess.run([
                'ffmpeg', '-y',
                '-i', audio_path,
                '-t', str(self.max_slide_duration),
                '-acodec', 'copy',
                temp_path
            ], check=True)
            
            os.replace(temp_path, audio_path)
        
        return audio_path

    def create_video(self, slides_content: List[Dict], audio_paths: List[str]) -> str:
        """Create high-quality video with professional transitions"""
        try:
            # First standardize all images
            print("Standardizing images...")
            for idx, slide in enumerate(slides_content):
                image_path = slide['image_path']
                try:
                    with Image.open(image_path) as img:
                        # Convert to RGB
                        if img.mode in ('RGBA', 'P', 'LA'):
                            background = Image.new('RGB', img.size, (255, 255, 255))
                            if img.mode == 'P':
                                img = img.convert('RGBA')
                            background.paste(img, mask=img.split()[3] if img.mode == 'RGBA' else None)
                            img = background
                        elif img.mode != 'RGB':
                            img = img.convert('RGB')
                        
                        # Standardize size to 1920x1080 while maintaining aspect ratio
                        target_width = 1920
                        target_height = 1080
                        
                        # Calculate resize dimensions
                        width, height = img.size
                        aspect = width / height
                        target_aspect = target_width / target_height
                        
                        if aspect > target_aspect:
                            # Image is wider than target
                            new_width = target_width
                            new_height = int(target_width / aspect)
                        else:
                            # Image is taller than target
                            new_height = target_height
                            new_width = int(target_height * aspect)
                        
                        # Resize image
                        img = img.resize((new_width, new_height), Image.LANCZOS)
                        
                        # Create new image with padding
                        new_img = Image.new('RGB', (target_width, target_height), (255, 255, 255))
                        paste_x = (target_width - new_width) // 2
                        paste_y = (target_height - new_height) // 2
                        new_img.paste(img, (paste_x, paste_y))
                        
                        # Save standardized image
                        standardized_path = os.path.join(self.image_dir, f"standardized_{idx}.jpg")
                        new_img.save(standardized_path, 'JPEG', quality=95)
                        slides_content[idx]['image_path'] = standardized_path
                        print(f"✓ Standardized image {idx} to 1920x1080")
                        
                except Exception as e:
                    print(f"✗ Error standardizing image {idx}: {str(e)}")
                    return None

            # Convert audio files to WAV
            wav_paths = []
            for idx, audio_path in enumerate(audio_paths):
                wav_path = os.path.join(self.temp_dir, f"audio_{idx}.wav")
                subprocess.run([
                    'ffmpeg', '-y',
                    '-i', audio_path,
                    '-acodec', 'pcm_s16le',
                    '-ar', '44100',
                    '-ac', '2',  # Force stereo
                    wav_path
                ], check=True)
                wav_paths.append(wav_path)

            # Combine audio files
            combined_audio = os.path.join(self.temp_dir, "combined_audio.wav")
            audio_list = os.path.join(self.temp_dir, "audio_list.txt")
            
            with open(audio_list, 'w', encoding='utf-8', newline='\n') as f:
                for wav_path in wav_paths:
                    escaped_path = os.path.abspath(wav_path).replace('\\', '/')
                    f.write(f"file '{escaped_path}'\n")
            
            subprocess.run([
                'ffmpeg', '-y',
                '-f', 'concat',
                '-safe', '0',
                '-i', audio_list,
                '-c:a', 'pcm_s16le',
                '-ar', '44100',
                '-ac', '2',  # Force stereo
                combined_audio
            ], check=True)
            
            # Create video input file
            input_file = os.path.join(self.temp_dir, "input.txt")
            print(f"\nProcessing {len(slides_content)} slides and {len(wav_paths)} audio files")
            
            with open(input_file, 'w', encoding='utf-8', newline='\n') as f:
                for idx, (slide, audio) in enumerate(zip(slides_content, wav_paths)):
                    duration = self._get_audio_duration(audio)
                    image_path = os.path.abspath(slide['image_path'])
                    
                    print(f"Processing slide {idx}:")
                    print(f"  Image path: {image_path}")
                    print(f"  Duration: {duration}")
                    
                    if not os.path.exists(image_path):
                        print(f"  ⚠️ Warning: Image file not found: {image_path}")
                        continue
                    
                    escaped_path = image_path.replace('\\', '/')
                    f.write(f"file '{escaped_path}'\n")
                    # Add small buffer to duration
                    adjusted_duration = duration + 0.5
                    f.write(f"duration {adjusted_duration:.6f}\n")
            
            # Debug: Print input file contents
            print("\nInput file contents:")
            with open(input_file, 'r', encoding='utf-8') as f:
                print(f.read())
            
            # Create final video
            output_path = os.path.join(
                self.output_dir,
                f"images_video_{int(time.time())}.mp4"
            )
            
            # Modified FFmpeg command with better error handling and filtering
            subprocess.run([
                'ffmpeg', '-y',
                '-f', 'concat',
                '-safe', '0',
                '-i', input_file,
                '-i', combined_audio,
                '-c:v', 'libx264',
                '-preset', 'slow',
                '-crf', '23',
                '-c:a', 'aac',
                '-b:a', '192k',
                '-pix_fmt', 'yuv420p',
                '-movflags', '+faststart',
                '-vf', 'format=yuv420p,fps=24',  # Simplified video filter
                '-shortest',
                output_path
            ], check=True)
            
            print("✓ Created high-quality video from images")
            return output_path
            
        except subprocess.CalledProcessError as e:
            print(f"✗ FFmpeg error: {e.output if hasattr(e, 'output') else str(e)}")
            raise
        except Exception as e:
            print(f"✗ Error creating video: {str(e)}")
            raise

    def _get_audio_duration(self, audio_path: str) -> float:
        """Get duration of audio file using FFprobe"""
        try:
            result = subprocess.run([
                'ffprobe',
                '-v', 'error',
                '-show_entries', 'format=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1',
                audio_path
            ], capture_output=True, text=True, check=True)
            
            duration = float(result.stdout.strip())
            return duration
        except subprocess.CalledProcessError as e:
            print(f"✗ FFprobe error: {str(e)}")
            return 3.0  # Default duration if unable to get actual duration
        except Exception as e:
            print(f"✗ Error getting audio duration: {str(e)}")
            return 3.0

    def cleanup(self):
        """Clean up temporary files after processing"""
        try:
            if os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
                print("✓ Temporary files cleaned up")
        except Exception as e:
            print(f"✗ Error during cleanup: {str(e)}")
import random
import re
import pdfplumber
import asyncio
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from img_gen import RealisticImageGenerator
from get_prompts import get_image_guide_prompt

from providers.image_factory import ImageFactory
from providers.image_provider import ImageProvider

def create_faiss_index(embeddings):
    dimension = embeddings.shape[1]  # Embedding dimension
    index = faiss.IndexFlatL2(dimension)  # L2 distance
    index.add(embeddings)
    return index

def extract_prompt(raw):
    print(f"Extracting prompt from raw text: {raw[:100]}...")
    content = re.search(r'<<(.+?)>>', raw)
    result = content.group(1) if content else raw
    print(f"Extracted prompt: {result[:100]}...")
    return result

def extract_items(input_string):
    print(f"Extracting items from: {input_string[:100]}...")
    # Find the text inside the << >>
    content = re.search(r'<<(.+?)>>', input_string)

    if content:
        content = content.group(1)
        print(f"Found content inside <<>>: {content[:100]}...")
    else:
        print("No content found inside <<>> markers")
        return []

    # Split the content by the | separator and remove whitespace
    items = [item.strip() for item in content.split('|')]
    
    # Remove the quotes from each item
    items = [re.sub(r'^"|"$', '', item) for item in items]
    
    print(f"Extracted {len(items)} items: {items}")
    return items

def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file."""
    print(f"Extracting text from PDF: {pdf_path}")
    extracted_text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            print(f"Successfully opened PDF with {len(pdf.pages)} pages")
            for i, page in enumerate(pdf.pages):
                print(f"Processing page {i+1}/{len(pdf.pages)}")
                page_text = page.extract_text()
                extracted_text += page_text
                print(f"Page {i+1} extracted {len(page_text)} characters")
    except Exception as e:
        print(f"Error reading PDF: {e}")
    
    print(f"Total extracted text: {len(extracted_text)} characters")
    return extracted_text

def chunk_text(text, chunk_size=500, overlap=100):
    print(f"Chunking text of {len(text)} characters with chunk_size={chunk_size}, overlap={overlap}")
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunks.append(text[i:i + chunk_size])
    
    print(f"Created {len(chunks)} chunks")
    return chunks

def generate_embeddings(chunks):
    print(f"Generating embeddings for {len(chunks)} chunks")
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')  # Offline embedding model
    print("SentenceTransformer model loaded")
    
    embeddings = model.encode(chunks)
    print(f"Generated embeddings with shape: {embeddings.shape}")
    return embeddings

def encode_query(query):
    print(f"Encoding query: {query}")
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')  # Offline embedding model
    embeddings = model.encode([query])
    print(f"Query encoded with shape: {embeddings.shape}")
    return embeddings

def query_faiss_index(index, query_embedding, k=5):
    print(f"Querying FAISS index with k={k}")
    distances, indices = index.search(query_embedding, k)  # Top k results
    print(f"Query results - indices: {indices}, distances: {distances}")
    return indices

def retrieve_pdf_context(faiss_index, query_embedding, chunk_texts, top_k=5):
    print(f"Retrieving PDF context with top_k={top_k}")
    indices = query_faiss_index(faiss_index, query_embedding, k=top_k)
    
    selected_chunks = [chunk_texts[i] for i in indices[0]]
    print(f"Selected chunks indices: {indices[0]}")
    
    context = " ".join(selected_chunks)
    print(f"Retrieved context with {len(context)} characters")
    return context

def build_prompt(prompt, context):
    if context:
        print(f"Building prompt with context of {len(context)} characters")
        prefix = "Given the following context -- Start of context -- \n: " + context + " -- End of context--\n\n"
        return prefix + prompt
    else:
        print("Building prompt without context")
        return prompt

def build_image_prompt(slide_content, image_provider='ai'):
    """
    Generates a structured prompt for image generation based on slide content.
    
    Args:
        slide_content: Content of the slide to base the image on
        image_provider: Type of image provider ('ai', 'web', or 'local')
    
    Returns:
        A properly formatted prompt for the specific image provider
    """
    print(f"Building image prompt for slide content with provider: {image_provider}")
    
    final_prompt = ""
    
    if image_provider == 'web':
        # Direct, literal web search prompt
        prefix = """Extract the most important named entities (people, countries, organizations, events) 
        from this slide content. Then create a LITERAL web search query that would find images showing these 
        specific entities. Include full names of people, places, or events.
        
        For example:
        - For content about US-China trade relations, use: <<President Biden President Xi meeting>>
        - For content about climate change impacts, use: <<flooding coastal cities climate change>>
        - For content about tech CEOs, use: <<Elon Musk Tim Cook Sundar Pichai>>
        
        The query must be specific, literal, and use proper names when mentioned. Never use metaphors or 
        artistic descriptions. Output only the search query.
        """
        
        output_guide = """
        Output ONLY the literal search query within << >> tags.
        Example: <<Modi Trump diplomatic meeting>>
        """
    
    elif image_provider == 'local':
        # Local image library prompt (unchanged)
        prefix = """You are an expert in creating effective image library search queries.
        Your task is to generate a concise, categorized search query based on the given slide content
        that will match relevant images from a local image library.
        
        Key Rules:
        - Generate a search query with 3-5 keywords that represent image categories or tags
        - Use common, general categories that would likely exist in an image library
        - Focus on subject matter, mood, and composition rather than specific details
        - Separate keywords with commas for better matching
        - NO explanations or variations - ONLY output the final search query
        """
        
        output_guide = """
        -- Start of Output Guide -- 
        - Output ONLY the final search keywords
        - Format must follow:  <<keyword1, keyword2, keyword3>>
        Example: <<business, teamwork, office, professional>>
        -- End of Output Guide -- 
        """
    
    else:  # Default 'ai' provider
        # AI art generation prompt (unchanged)
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
        guide = get_image_guide_prompt()
        print(f"Retrieved image guide prompt: {len(guide)} characters")
        
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
            
        print(f"Built AI image prompt with total length: {len(final_prompt)} characters")
        return final_prompt

    # For web and local providers, build a simpler prompt
    slide_section = f"\n-- Slide Content --\n{slide_content}\n-- End of Slide Content --\n"
    final_prompt = prefix + slide_section + output_guide
    
    print(f"Built {image_provider} image prompt with total length: {len(final_prompt)} characters")
    return final_prompt

def data_gen(params):
    print(f"Generating presentation: {params.get('title', 'Untitled')}")

    # Initialize Gemini API
    # Note: You'll need to set your API key in your environment or config
    API_KEY = "AIzaSyB1U3NavO8CvkQ2pk0NFpEf_NKYJPCnAPk"  # Replace with actual API key or load from environment
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
    model = genai.GenerativeModel(
        model_name="gemini-1.5-pro",
        generation_config=generation_config
    )
    
    
    options = params['options']
    topic = params['topic']
    
    print(f"Processing topic: '{topic}' with options: {options}")
    
    context = None
    
    if 'context' in params:
        print(f"Context file provided: {params['context']}")
        texts = extract_text_from_pdf(params['context'])
        chunks = chunk_text(texts)
        print(f"Processing {len(chunks)} text chunks for embeddings")
        embeddings = generate_embeddings(chunks)
        indices = create_faiss_index(embeddings)
        encoded_query = encode_query(topic)
        context = retrieve_pdf_context(indices, encoded_query, chunks)
        print(f"Retrieved context with {len(context)} characters")
    else:
        print("No context file provided, proceeding without context")
    
    slide_data = {}
    
    print(f"Parsing context for query: {topic} [1]")
    
    # Function to use Gemini model instead of Ollama
    def invoke_gemini(prompt):
        print(f"Invoking Gemini with prompt of {len(prompt)} characters")
        try:
            response = model.generate_content(prompt)
            print(f"Received response from Gemini with {len(response.text)} characters")
            return response.text
        except Exception as e:
            print(f"Error invoking Gemini API: {e}")
            raise
    
    # Get title and subtitle
    print("Generating title and subtitle")
    title_prompt = build_prompt(f"""
    You are a text summarization and formatting specialized model that fetches relevant information, on the scale of (Concise, Moderate, Comprehensive, Extensive) keep the information size as {options['contentSize']} and clear.

    For the topic "{topic}" suggest a presentation title and a presentation subtitle it should be returned in the format:
    << "title" | "subtitle >>

    example :
    << "Ethics in Design" | "Integrating Ethics into Design Processes" >>
    """, context)
    
    title_response = invoke_gemini(title_prompt)
    print(f"Title prompt response: {title_response}")
    
    result = extract_items(title_response)
    
    if len(result) >= 2:
        slide_data['title'], slide_data['subtitle'] = result[0], result[1]
        print(f"Extracted title: '{slide_data['title']}' and subtitle: '{slide_data['subtitle']}'")
    else:
        print(f"Failed to extract both title and subtitle, got: {result}")
        slide_data['title'] = result[0] if result else "Presentation"
        slide_data['subtitle'] = result[1] if len(result) > 1 else "Subtitle"
    
    print("Building title and subtitle for parsed context [2]")
    print(f"Suggested title: {slide_data['title']} | Suggested subtitle: {slide_data['subtitle']} [3]")
    
    print("Building topics for the context [4]")
    
    # Get slide topics
    topics_prompt = build_prompt(f"""
    You are a text summarization and formatting specialized model that fetches relevant information, on the scale of (Concise, Moderate, Comprehensive, Extensive) keep the information size as {options['contentSize']} and clear.
       
    For the presentation titled "{slide_data['title']}" and with subtitle "{slide_data['subtitle']}" for the topic "{topic}"
    Write a table of contents containing the title of each slide for a {options['slideCount']} slide presentation
    It should be of the format :
    << "slide1" | "slide2" | "slide3" | ... | "slide{options['slideCount']}" >>

    example :
    << "Introduction to Design Ethics" | "User-Centered Design" | "Transparency and Honesty" | "Data Privacy and Security" | "Accessibility and Inclusion" | "Social Impact and Sustainability" | "Ethical AI and Automation" | "Collaboration and Professional Ethics" >>          
    """, context)
    
    topics_response = invoke_gemini(topics_prompt)
    print(f"Topics prompt response: {topics_response}")
    
    result = extract_items(topics_response)
    
    if result:
        slide_data['overview'] = result
        print(f"Generated {len(result)} topics: {result}")
    else:
        print("Failed to extract topics, using default")
        slide_data['overview'] = [f"Slide {i+1}" for i in range(options['slideCount'])]
    
    print("Topics built successfully => [5]" + str(slide_data['overview']))
    print("Initiating PPT build of " + str(options['slideCount']) + " slides [6]")
    
    slide_data['content'] = []
    
    
    has_image = options['hasImage']
    
    
    provider_config = {}
    image_provider = None
    
    if has_image:
        image_provider = options['imageProvider'].lower()
        if image_provider == 'ai':
            # AI image generation config
            provider_config = {
                'save_dir': "uploads/generated_image",
                'device': 'cuda'
            }
        elif image_provider == 'web':
            # Web image fetching config
            provider_config = {
                'save_dir': "uploads/generated_image"
            }
        elif image_provider == 'local':
            # Local image library config
            provider_config = {
                'save_dir': "uploads/generated_image",
                'image_library_path': params.get('imageLibraryPath', 'image_library')
            }
        elif image_provider == 'google':
            # Google image fetching config
            provider_config = {
                'save_dir': "uploads/generated_image",
                'api_key': "AIzaSyDdxB2ZCOoMB5ciwpSPBP3olsvLmGp8Wy0",
                'cx_id': "3539b4c3899364950"
            }
        else:
            print(f"⚠️ Unknown image source: {image_provider}, defaulting to AI generation")
            provider_config = {
                'save_dir': "uploads/generated_image",
                'device': 'cuda'
            }
        
        image_factory = ImageFactory.create_provider(image_provider, provider_config)
        
        if not image_factory:
            print(f"⚠️ Failed to create image provider for source: {image_provider}")
            # Fall back to RealisticImageGenerator for backward compatibility
            print("Falling back to RealisticImageGenerator")
            from img_gen import RealisticImageGenerator
            image_factory = RealisticImageGenerator()
            
            
    
    # Generate content for each slide
    for idx, subtopic in enumerate(slide_data['overview'], 1):
        print(f"\nSlide {idx}: {subtopic}")
        
        # Get draft content
        data_prompt = build_prompt(f"""
        You are a text summarization and formatting specialized model that fetches relevant information, on the scale of (Concise, Moderate, Comprehensive, Extensive) keep the information size as {options['contentSize']} and clear.
            
        For the presentation titled "{slide_data['title']}" and with subtitle "{slide_data['subtitle']}" for the topic "{topic}"
        Write the contents for a slide with the subtopic {subtopic}
        Write {options['pointCount']} points.
        """, context)
        
        print(f"Invoking Gemini for slide content draft")
        data_to_clean = invoke_gemini(data_prompt)
        print(f"Received draft content with {len(data_to_clean)} characters")
        
        # Format the content
        clean_prompt = build_prompt(f"""
        You are a text summarization and formatting specialized model that fetches relevant information, on the scale of (Concise, Moderate, Comprehensive, Extensive) keep the information size as {options['contentSize']} and clear also formats it into user specified formats.
        Given below is a text draft for a presentation slide containing {options['pointCount']} points, extract {options['pointCount']} sentences and format it as:
                    
        << "point1" | "point2" | "point3"  | ... | "point{options['pointCount']}" >>
        
         example :
        << "Foster a collaborative and inclusive work environment." | "Respect intellectual property rights and avoid plagiarism." | "Uphold professional standards and codes of ethics." | "Be open to feedback and continuous learning." >>

        
        -- Beginning of the text --
        {data_to_clean}
        -- End of the text --         
        """, context)
        
        print(f"Invoking Gemini for content cleaning and formatting")
        cleaned_data = invoke_gemini(clean_prompt)
        print(f"Received cleaned content: {cleaned_data}")
        
        # Extract points from cleaned data
        points = extract_items(cleaned_data)
        print(f"Extracted {len(points)} points for slide")
        
        # Add slide to data
        item = {
            'title': subtopic,
            'points': points,
        }
        
        if has_image:
            print(f"\nProcessing image for slide: {subtopic}")
            
            if image_provider in ['web', 'google', 'stock']:
                image_prompt = topic + " " + item['points'][random.randint(0, len(item['points']) - 1)]
            else:
                image_prompt_text = ImageProvider.build_image_prompt(cleaned_data, image_provider=image_provider)
                raw_image_prompt = invoke_gemini(image_prompt_text)
                image_prompt = extract_prompt(raw_image_prompt)

            try:
                filepaths = image_factory.get_images([image_prompt])
                item['image'] = filepaths[0]
                print(f"✓ Image added")
            except Exception as e:
                print(f"✗ Image failed: {str(e)}")
    
        
        
        slide_data['content'].append(item)
        print(f"Added slide {idx} to presentation data")
    
    print("\n✓ Presentation generated successfully")
    return slide_data
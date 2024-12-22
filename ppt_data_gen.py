import re
from langchain.llms import Ollama
import pdfplumber

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

def create_faiss_index(embeddings):
    dimension = embeddings.shape[1]  # Embedding dimension
    index = faiss.IndexFlatL2(dimension)  # L2 distance
    index.add(embeddings)
    return index

def extract_items(input_string):
    print(input_string)
    # Find the text inside the << >>
    content = re.search(r'<<(.+?)>>', input_string)

    if content:
        content = content.group(1)
    else:
        return []

    # Split the content by the | separator and remove whitespace
    items = [item.strip() for item in content.split('|')]

    # Remove the quotes from each item
    items = [re.sub(r'^"|"$', '', item) for item in items]

    return items

def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file."""
    extracted_text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                extracted_text += page.extract_text()
    except Exception as e:
        print(f"Error reading PDF: {e}")
    return extracted_text

def chunk_text(text, chunk_size=500, overlap=100):
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunks.append(text[i:i + chunk_size])
    return chunks

def generate_embeddings(chunks):
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')  # Offline embedding model
    embeddings = model.encode(chunks)
    return embeddings

def encode_query(query):
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')  # Offline embedding model
    embeddings = model.encode([query])
    return embeddings

def query_faiss_index(index, query_embedding, k=5):
    distances, indices = index.search(query_embedding, k)  # Top k results
    return indices

def retrieve_pdf_context(faiss_index, query_embedding, chunk_texts, top_k=5):
    indices = query_faiss_index(faiss_index, query_embedding, k=top_k)
    return " ".join([chunk_texts[i] for i in indices[0]])

def build_prompt(prompt, context):
    if context:
        prefix = "Given the following context -- Start of context -- \n: " + context + " -- End of context--\n\n"
        return prefix + prompt
    else:
        return prompt

def data_gen(params): # Chinese Food
    llm = Ollama(model="llama3.1:8b",
                 temperature="0.4")
    
    options = params['options']
    topic = params['topic']

    context = None
    
    if 'context' in params:
        texts = extract_text_from_pdf(params['context'])
        chunks = chunk_text(texts)
        embeddings = generate_embeddings(chunks)
        indices = create_faiss_index(embeddings)
        encoded_query = encode_query(topic)
        context = retrieve_pdf_context(indices, encoded_query, chunks)
    
    slide_data = {}


    print(f"Parsing context for query: {topic} [1]")
        
    result = extract_items(llm.invoke(build_prompt(f"""
    You are a text summarization and formatting specialized model that fetches relevant information, on the scale of (Concise, Moderate, Comprehensive, Extensive) keep the information size as {options['contentSize']} and clear.

    For the topic "{topic}" suggest a presentation title and a presentation subtitle it should be returned in the format:
    << "title" | "subtitle >>

    example :
    << "Ethics in Design" | "Integrating Ethics into Design Processes" >>
    """, context)))

    slide_data['title'], slide_data['subtitle'] = result[0], result[1]

    print("Building title and subtitle for parsed context [2]")
    print(f"Suggested title: {slide_data['title']} | Suggested subtitle: {slide_data['subtitle']} [3]")

    print("Building topics for the context [4]")


    result = extract_items(llm.invoke(build_prompt(f"""
    You are a text summarization and formatting specialized model that fetches relevant information, on the scale of (Concise, Moderate, Comprehensive, Extensive) keep the information size as {options['contentSize']} and clear.
       
    For the presentation titled "{slide_data['title']}" and with subtitle "{slide_data['subtitle']}" for the topic "{topic}"
    Write a table of contents containing the title of each slide for a {options['slideCount']} slide presentation
    It should be of the format :
    << "slide1" | "slide2" | "slide3" | ... | "slide{options['slideCount']}" >>

    example :
    << "Introduction to Design Ethics" | "User-Centered Design" | "Transparency and Honesty" | "Data Privacy and Security" | "Accessibility and Inclusion" | "Social Impact and Sustainability" | "Ethical AI and Automation" | "Collaboration and Professional Ethics" >>          
    """, context)))

    slide_data['overview'] = result

    print("Topics built succesfully => [5]" + str(result))
    print("Initiating PPT build of " + str(options['slideCount']) + " slides [6]")

    slide_data['content'] = []

    for subtopic in slide_data['overview']:
        print(f"Generating slide content for: {subtopic}")
        
        data_to_clean = llm.invoke(build_prompt(f"""
        You are a text summarization and formatting specialized model that fetches relevant information, on the scale of (Concise, Moderate, Comprehensive, Extensive) keep the information size as {options['contentSize']} and clear.
            
        For the presentation titled "{slide_data['title']}" and with subtitle "{slide_data['subtitle']}" for the topic "{topic}"
        Write the contents for a slide with the subtopic {subtopic}
        Write {options['pointCount']} points.
        """, context))



        cleaned_data = llm.invoke(build_prompt(f"""
        You are a text summarization and formatting specialized model that fetches relevant information, on the scale of (Concise, Moderate, Comprehensive, Extensive) keep the information size as {options['contentSize']} and clear also formats it into user specified formats.
        Given below is a text draft for a presentation slide containing {options['pointCount']} points, extract {options['pointCount']} sentences and format it as:
                    
        << "point1" | "point2" | "point3"  | ... | "point{options['pointCount']}" >>
        
         example :
        << "Foster a collaborative and inclusive work environment." | "Respect intellectual property rights and avoid plagiarism." | "Uphold professional standards and codes of ethics." | "Be open to feedback and continuous learning." >>

        
        -- Beginning of the text --
        {data_to_clean}
        -- End of the text --         
        """, context))
        
        item = {
            'title':subtopic,
            'points': extract_items(cleaned_data)
        }

        print(item)

        slide_data['content'].append(item)
        

    print("GenPT Build finished, download the result from the UI :)")
    return slide_data

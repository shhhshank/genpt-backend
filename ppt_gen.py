from pptx import Presentation
import io
import re
import inspect
import copy
import time

from pptx.dml.color import _NoneColor

def sanitize_string(input_str):
    sanitized = re.sub(r"[^A-Za-z0-9_.-]", "", input_str)
    sanitized = re.sub(r"\.{2,}", ".", sanitized)
    sanitized = re.sub(r"^[^A-Za-z0-9]+", "", sanitized)
    sanitized = re.sub(r"[^A-Za-z0-9]+$", "", sanitized)
    sanitized = sanitized[:63] if len(sanitized) > 63 else sanitized.ljust(3, "_")
    return sanitized



def duplicate_slide(pres, template, idx):
    # Add a new slide
    copied_slide = pres.slides.add_slide(template.slide_layout)
    
    # Delete the existing shapes that are part of the layout
    for shp in copied_slide.shapes:
        copied_slide.shapes.element.remove(shp.element)        
    
    # Perform a deep copy of the shapes from the template
    for shp in template.shapes:
        if "Picture" in shp.name:
            img = io.BytesIO(shp.image.blob)
            copied_slide.shapes.add_picture(image_file = img,
                                            left = shp.left,
                                            top = shp.top,
                                            width = shp.width,
                                            height = shp.height)
        else:
            el = shp.element
            newel = copy.deepcopy(el)
            copied_slide.shapes._spTree.insert_element_before(newel, 'p:extLst')
    slides = pres.slides._sldIdLst  
    slides.insert(idx, slides[-1])  # Move the last slide to the specified index

    return copied_slide
    


def duplicate_run_in_paragraph(paragraph, run_to_duplicate):

    # Create a new Run in the same Paragraph
    new_run = paragraph.add_run()


    # Copy font properties (if set)
    try:
        if run_to_duplicate.font.size:
            new_run.font.size = run_to_duplicate.font.size
        if run_to_duplicate.font.bold is not None:
            new_run.font.bold = run_to_duplicate.font.bold
        if run_to_duplicate.font.italic is not None:
            new_run.font.italic = run_to_duplicate.font.italic

        if run_to_duplicate.font.color and run_to_duplicate.font.color.rgb:
                new_run.font.color.rgb = run_to_duplicate.font.color.rgb
        if run_to_duplicate.font.name:
            new_run.font.name = run_to_duplicate.font.name
    except:
        pass
        


    return new_run

def duplicate_bullet(paragraph, shape):

    # Duplicate the paragraph within the same shape
    new_paragraph = shape.text_frame.add_paragraph()
    new_paragraph.text = paragraph.text
    new_paragraph.font.size = paragraph.font.size
    new_paragraph.font.bold = paragraph.font.bold
    new_paragraph.font.italic = paragraph.font.italic
    new_paragraph.font.underline = paragraph.font.underline
    new_paragraph.level = paragraph.level
    new_paragraph.bullet = True
    
    print(inspect.getmembers(paragraph))


    return new_paragraph

def replace_text(slide, search_text, replace_text):

    # Create a case-insensitive pattern for search_text
    pattern = re.compile(re.escape(search_text), re.IGNORECASE)

    for shape in slide.shapes:
        # Check if the shape has a text frame (contains text)
        if not shape.has_text_frame:
            continue
        
        # Loop through paragraphs and runs within the text frame
        for paragraph in shape.text_frame.paragraphs:
            for run in paragraph.runs:
                # Replace text using the case-insensitive pattern
                if re.search(pattern, run.text):
                    run.text = re.sub(pattern, replace_text, run.text)
    
    return slide

def get_view_from_text(slide, search_text):
    for shape in slide.shapes:
        # Check if the shape has a text frame (contains text)
        if not shape.has_text_frame:
            continue
        
        # Loop through paragraphs and runs within the text frame
        for paragraph in shape.text_frame.paragraphs:
            print(paragraph.text)
            if paragraph.text == search_text:
                return paragraph
    
    return None


def fill_title(ppt, title, subtitle):
    slide = ppt.slides[0]
    replace_text(slide, "[title]", title)
    replace_text(slide, "[subtitle]", subtitle)

def fill_overview(ppt, overview):
    title_search = "[slide_title_i]"
    slide = ppt.slides[1]
    paragraph = get_view_from_text(slide, title_search)

    if paragraph:
        run_to_dup = paragraph.runs[0]
        paragraph.text = ""
        idx = -1
        for item in overview:
            idx += 1
            r_dup = duplicate_run_in_paragraph(paragraph, run_to_dup)
            r_dup.text = item
            if (idx < len(overview) - 1):
                r_dup.text += '\n'
    else:
        print("No paragraph found for: " + title_search)    

def fill_content(ppt, content):
    slide = ppt.slides[2]
    
    for i in range(len(content) - 1):
        duplicate_slide(ppt, slide, i + 3)
    
    idx = 2
    for item in content:
        curr_slide = ppt.slides[idx]
        title_para = get_view_from_text(curr_slide, "[slide_top_title_i]")
        if title_para:
            title_para.text = item['title']
        
        bullet_para = get_view_from_text(curr_slide, "[slide_i_point_j]")

        if bullet_para:
            run_to_dup = bullet_para.runs[0]
            bullet_para.text = ""
            pt_idx = -1
            for point in item['points']:
                pt_idx += 1
                r_dup = duplicate_run_in_paragraph(bullet_para, run_to_dup)
                r_dup.text = point
                if (pt_idx < len(item['points']) - 1):
                    r_dup.text += '\n'
        else:
            print("No paragraph found for: [slide_i_point_j]")

        idx += 1
    

def ppt_gen(template_path, slide_data):
    ppt = Presentation(template_path)

    title = slide_data['title']
    subtitle = slide_data['subtitle']
    overview = slide_data['overview']
    content = slide_data['content']

    fill_title(ppt, title, subtitle)
    fill_overview(ppt, overview)
    fill_content(ppt, content)



    path = "uploads/generated_ppt/" + str(time.time() * 1000) +".pptx"
    ppt.save(path)


    return  'http://127.0.0.1:8080/' + path 

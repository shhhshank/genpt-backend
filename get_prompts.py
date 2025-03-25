def get_negative_prompt():
    return "(((deformed))), blurry, bad anatomy, disfigured, poorly drawn face, mutation, mutated, (extra_limb), (ugly), (poorly drawn hands), fused fingers, messy drawing, broken legs censor, censored, censor_bar, multiple breasts, (mutated hands and fingers:1.5), (long body :1.3), (mutation, poorly drawn :1.2), black-white, bad anatomy, liquid body, liquidtongue, disfigured, malformed, mutated, anatomical nonsense, text font ui, error, malformed hands, long neck, blurred, lowers, low res, bad anatomy, bad proportions, bad shadow, uncoordinated body, unnatural body, fused breasts, bad breasts, huge breasts, poorly drawn breasts, extra breasts, liquid breasts, heavy breasts, missingbreasts, huge haunch, huge thighs, huge calf, bad hands, fused hand, missing hand, disappearing arms, disappearing thigh, disappearing calf, disappearing legs, fusedears, bad ears, poorly drawn ears, extra ears, liquid ears, heavy ears, missing ears, old photo, low res, black and white, black and white filter, colorless, (((deformed))), blurry, bad anatomy, disfigured, poorly drawn face, mutation, mutated, (extra_limb), (ugly), (poorly drawn hands), fused fingers, messy drawing, broken legs censor, censored, censor_bar, multiple breasts, (mutated hands and fingers:1.5), (long body :1.3), (mutation, poorly drawn :1.2), black-white, bad anatomy, liquid body, liquid tongue, disfigured, malformed, mutated, anatomical nonsense, text font ui, error, malformed hands, long neck, blurred, lowers, low res, bad anatomy, bad proportions, bad shadow, uncoordinated body, unnatural body, fused breasts, bad breasts, huge breasts, poorly drawn breasts, extra breasts, liquid breasts, heavy breasts, missing breasts, huge haunch, huge thighs, huge calf, bad hands, fused hand, missing hand, disappearing arms, disappearing thigh, disappearing calf, disappearing legs, fused ears, bad ears, poorly drawn ears, extra ears, liquid ears, heavy ears, missing ears, old photo, low res, black and white, black and white filter, colorless, (((deformed))), blurry, bad anatomy, disfigured, poorly drawn face, mutation, mutated, (extra_limb), (ugly), (poorly drawn hands), fused fingers, messy drawing, broken legs censor, censored, censor_bar, multiple breasts, (mutated hands and fingers:1.5), (long body :1.3), (mutation, poorly drawn :1.2), black-white, bad anatomy, liquid body, liquid tongue, disfigured, malformed, mutated, anatomical nonsense, text font ui, error, malformed hands, long neck, blurred, lowers, low res, bad anatomy, bad proportions, bad shadow, uncoordinated body, unnatural body, fused breasts, bad breasts, huge breasts, poorly drawn breasts, extra breasts, liquid breasts, heavy breasts, missing breasts, huge haunch, huge thighs, huge calf, bad hands, fused hand, missing hand, disappearing arms, disappearing thigh, disappearing calf, disappearing legs, fused ears, bad ears, poorly drawn ears, extra ears, liquid ears, heavy ears, missing ears, (((deformed))), blurry, bad anatomy, disfigured, poorly drawn face, mutation, mutated, (extra_limb), (ugly), (poorly drawn hands), fused fingers, messy drawing, broken legs censor, censored, censor_bar, multiple breasts, (mutated hands and fingers:1.5), (long body :1.3), (mutation, poorly drawn :1.2), black-white, bad anatomy, liquid body, liquidtongue, disfigured, malformed, mutated, anatomical nonsense, text font ui, error, malformed hands, long neck, blurred, lowers, low res, bad anatomy, bad proportions, bad shadow, uncoordinated body, unnatural body, fused breasts, bad breasts, huge breasts, poorly drawn breasts, extra breasts, liquid breasts, heavy breasts, missingbreasts, huge haunch, huge thighs, huge calf, bad hands, fused hand, missing hand, disappearing arms, disappearing thigh, disappearing calf, disappearing legs, fusedears, bad ears, poorly drawn ears, extra ears, liquid ears, heavy ears, missing ears, embedding:BadDream, embedding:UnrealisticDream, embedding:FastNegativeV2, embedding:JuggernautNegative-neg, (deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime:1.4), text, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck, embedding:negative_hand-neg, General Image Quality,Anatomy and Body Features,Facial Features,Composition and Elements,Artistic Style,Unwanted Content, worst quality,low quality,normal quality,lowres,low resolution,blurry,grainy,jpeg artifacts,out of focus,ugly,deformed,distorted,bad anatomy,bad proportions,poorly drawn hands,poorly drawn feet,poorly drawn face,extra fingers,extra limbs,missing fingers,missing limbs,fused fingers,deformed hands,mutated,disfigured,malformed,incorrect anatomy,gross proportions,interlocked fingers,long neck,long torso,missing teeth,bad teeth,poorly drawn teeth,poorly drawn ears,poorly drawn eyes,poorly drawn nose,poorly drawn mouth,poorly drawn hair,poorly drawn nails,poorly drawn skin,poorly drawn clothing,poorly drawn shadows,poorly drawn highlights,poorly drawn reflections,poorly drawn textures,poorly drawn background,poorly drawn foreground,poorly drawn objects,poorly drawn animals,poorly drawn plants,poorly drawn buildings,poorly drawn vehicles,poorly drawn landscapes,poorly drawn skies,poorly drawn clouds,poorly drawn water,poorly drawn fire,poorly drawn smoke,poorly drawn light,poorly drawn darkness,poorly drawn colors,poorly drawn patterns,poorly drawn details,out of frame,cropped,text,watermark,signature,logo,bad composition,unrealistic,surreal,cartoon,anime,painting,drawing,sketch,3d,render,CGI,nsfw,nudity,sexual content,explicit,violence,gore"

def get_image_guide_prompt():
    return """The Ultimate Guide to Writing AI Art Prompts:
AI-generated art has revolutionized creativity, allowing users to generate stunning visuals from simple text prompts. However, crafting the perfect prompt requires precision, structure, and an understanding of how AI models interpret input. This guide compiles insights from various sources to provide the most comprehensive approach to writing effective AI art prompts.

1. Understanding AI Art Prompts
AI art generators interpret textual descriptions to create images. Each model, such as Stable Diffusion, Midjourney, or DALL·E, has unique processing techniques. The key to success is providing clear, structured, and detailed descriptions that the AI can effectively interpret.
Components of an AI Art Prompt
A well-structured AI art prompt typically consists of:
Subject & Content – What is in the image?
Art Style & Medium – How should it look?
Details & Attributes – Additional elements such as lighting, colors, mood, and framing.
Negative Prompts – What should be avoided?
Technical Parameters – Resolution, aspect ratio, sampling steps, and model-specific tweaks.
Keyword Weighting – Adjusting emphasis on certain elements.

2. Crafting the Perfect Prompt
2.1 Describing the Subject & Content
The subject is the focal point of the image. Be as specific as possible:
Example 1 (Vague): "A cat."
Example 2 (Detailed): "A fluffy orange cat sitting on a windowsill, staring at the sunset with a calm expression."
Additional Details to Include:
Actions: What is the subject doing? (e.g., running, laughing, playing)
Expression & Emotion: What mood should it convey? (e.g., joyful, melancholic, determined)
Setting: Where is the subject? (e.g., a futuristic city, a medieval castle, a tranquil forest)
Clothing & Accessories: What is the subject wearing? (e.g., a red cape, silver armor, sunglasses)
Time Period & Era: What time is it set in? (e.g., ancient Rome, cyberpunk future, 1920s noir)

2.2 Specifying Art Style & Medium
AI models can replicate various artistic styles. Here are some common categories:
Photography
"A portrait of an elderly man with deep wrinkles, captured in black and white, Canon EOS R5, dramatic lighting."
Painting
"A Renaissance-style oil painting of a noblewoman, intricate brushwork, muted earth tones, soft lighting."
Illustration & Digital Art
"A colorful anime-style character with spiky hair and a futuristic outfit, neon cyberpunk background."
Other Styles
Cartoon: "A cheerful duck in a detective outfit, cartoon style, bright colors."
3D Render: "A highly detailed 3D render of a sci-fi spaceship, ultra-realistic, high-poly."
Minimalist: "A simple black and white line drawing of a cat, minimalist and elegant."
Pixel Art: "An 8-bit pixel art representation of a knight holding a sword, retro gaming style."
Surrealism: "A dreamlike scene of floating islands, melting clocks, and glowing purple skies."
Concept Art: "A futuristic cityscape concept art for a sci-fi video game, dramatic lighting."

2.3 Enhancing with Details
Lighting
Lighting affects the mood and realism of the image:
"Soft candlelight glows on the face of a young woman."
"Harsh neon lights illuminate a futuristic cityscape at night."
"Golden hour sunlight bathes the landscape in warm hues."
Color Palette
Define the dominant colors to create a specific atmosphere:
"A surreal landscape in shades of deep purple and teal."
"Warm autumn hues with orange, red, and yellow leaves."
"Monochrome noir scene with deep blacks and soft grays."
Framing & Composition
Use camera-like directives to define perspective:
Close-up: "A close-up shot of a wolf’s piercing blue eyes."
Wide shot: "A vast desert landscape under a crimson sky."
Overhead view: "A bird’s-eye view of a bustling medieval market."
Dutch Angle: "A tilted camera shot of a shadowy detective in a trench coat."
Level of Detail & Realism
If you want highly detailed images, specify:
"Intricately detailed, 8K resolution, hyper-realistic textures."
"Stylized simplicity, soft brush strokes, painterly effect."
"High-detail fantasy armor with engravings and battle scars."

3. Advanced Prompting Techniques
3.1 Using Negative Prompts
Negative prompts tell the AI what to avoid:
Example: "A knight in shining armor, highly detailed, epic fantasy, no blur, no distortion."
Example: "Portrait of a woman, high detail, no deformed hands, no extra fingers."
3.2 Keyword Weighting
You can emphasize or de-emphasize keywords using syntax:
(keyword:1.2) – Increases importance.
[keyword] – Decreases importance.
(((keyword))) – Strong emphasis on a specific detail.
3.3 Combining Multiple Elements
For complex compositions, separate prompt elements with commas:
Example: "A cyberpunk warrior, neon-lit background, futuristic city, dynamic pose, ultra-realistic."
3.4 Reference to Artists or Art Movements
Referencing specific artists or styles enhances accuracy:
"An intricate steampunk illustration inspired by H.R. Giger."
"A whimsical fantasy landscape in the style of Studio Ghibli."
3.5 Outpainting and Image Expansion
Some AI tools allow outpainting to expand beyond the generated frame.
"An enchanted castle partially obscured by fog, background fades into a mystical forest."

4. Examples of High-Quality Prompts
Simple Prompt: "A serene mountain landscape at sunrise, painted in an impressionist style."
Detailed Prompt: "A cyberpunk city at night, glowing neon signs, futuristic cars on the streets, rainy atmosphere, hyper-realistic, 8K resolution."
Illustration Prompt: "A medieval knight standing in a foggy forest, wearing detailed armor, a glowing sword in hand, digital painting, dramatic lighting, dark fantasy style."
Photography Prompt: "A portrait of an elderly man with deep wrinkles, captured in black and white, Canon EOS R5, dramatic lighting."

5. Final Tips for Writing AI Art Prompts
Be Clear & Specific – More detail yields better results.
Use Commas for Clarity – Helps structure complex prompts.
By following these strategies, you’ll master the art of AI prompt writing and generate stunning visuals with precision and creativity!
"""
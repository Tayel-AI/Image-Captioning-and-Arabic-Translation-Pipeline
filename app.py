# Import necessary libraries and modules
from transformers import BlipProcessor, BlipForConditionalGeneration, MBartForConditionalGeneration, MBart50Tokenizer
from gtts import gTTS
from PIL import Image
import gradio as gr

# Pipeline Component 1: Image Captioning Model
class ImageToText:
    def __init__(self):
        """Initializes the BLIP model for image captioning."""
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        print("BLIP Image Captioning Model Loaded")

    def generate_caption(self, img):
        """Generates a caption for the given image."""
        inputs = self.processor(images=img, return_tensors="pt")
        generated_ids = self.model.generate(**inputs)
        caption = self.processor.decode(generated_ids[0], skip_special_tokens=True)
        return caption

# Pipeline Component 2: Arabic Translation Model (mBART)
class ArabicTranslator:
    def __init__(self):
        """Initializes the mBART model for English to Arabic translation."""
        self.tokenizer = MBart50Tokenizer.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
        self.model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
        print("mBART Arabic Translation Model Loaded")

    def translate(self, text):
        """Translates the given English text to Arabic."""
        inputs = self.tokenizer(text, return_tensors="pt", src_lang="en_XX")
        translated = self.model.generate(inputs["input_ids"], forced_bos_token_id=self.tokenizer.lang_code_to_id["ar_AR"])
        translated_text = self.tokenizer.batch_decode(translated, skip_special_tokens=True)[0]
        return translated_text

# Pipeline Component 3: Text-to-Speech Model (gTTS)
class TextToSpeech:
    def __init__(self, lang='ar'):
        """Initializes the Text-to-Speech system for Arabic."""
        self.lang = lang

    def generate_audio(self, text):
        """Generates audio from the given Arabic text."""
        tts = gTTS(text=text, lang=self.lang, slow=False)
        audio_file_path = 'output.mp3'
        tts.save(audio_file_path)
        return audio_file_path

# Main Pipeline Integration
class ImageToArabicSpeechPipeline:
    def __init__(self):
        """Initializes all pipeline components."""
        self.caption_model = ImageToText()
        self.translation_model = ArabicTranslator()
        self.tts_model = TextToSpeech()

    def process_image(self, img):
        """Processes the image, generates a caption, translates it to Arabic, and converts it to speech."""
        caption = self.caption_model.generate_caption(img)
        translated_text = self.translation_model.translate(caption)
        audio_file = self.tts_model.generate_audio(translated_text)
        return caption, translated_text, audio_file

# Gradio Interface Setup
def demo(image):
    """Function to be used in Gradio for processing the image and returning caption, translation, and audio."""
    img = Image.open(image)
    pipeline = ImageToArabicSpeechPipeline()
    caption, translated_text, audio_file = pipeline.process_image(img)
    return caption, translated_text, audio_file

# Define Gradio Interface
iface = gr.Interface(
    fn=demo,
    inputs=gr.Image(type="filepath"),
    outputs=[gr.Textbox(label="Caption"), gr.Textbox(label="Translated Text"), gr.Audio(label="Generated Speech")]
)

# Launch the Gradio Interface
iface.launch()

import torch
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import google.generativeai as genai
from PIL import Image, ImageDraw
from io import BytesIO
import pyttsx3
from googletrans import Translator
import pytesseract
import streamlit as st
import os

# --- Configuration for Google Generative AI ---
def configure_generative_ai(api_key):
    genai.configure(api_key=api_key)

# --- Object Detection Model ---
# Load object detection model
def load_object_detection_model():
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    return model

# Detect objects in the image
def detect_objects(image, object_detection_model, threshold=0.5):
    transform = transforms.Compose([transforms.ToTensor()])
    img_tensor = transform(image)
    predictions = object_detection_model([img_tensor])[0]

    filtered_boxes = [
        (box, label, score)
        for box, label, score in zip(predictions['boxes'], predictions['labels'], predictions['scores'])
        if score > threshold
    ]
    return filtered_boxes

# Draw bounding boxes on the image
def draw_boxes(image, predictions):
    draw = ImageDraw.Draw(image)
    for box, label, score in predictions:
        x1, y1, x2, y2 = box.tolist()
        draw.rectangle([x1, y1, x2, y2], outline="yellow", width=3)
        draw.text((x1, y1), f"Object ID {label.item()} ({score:.2f})", fill="black")
    return image

# --- Generative AI ---
def generate_scene_description(input_prompt, image_data):
    try:
        model = genai.GenerativeModel("gemini-1.5-pro")
        response = model.generate_content([input_prompt, image_data[0]])
        return response.text
    except Exception as e:
        return f"⚠ Error generating scene description: {str(e)}"

def generate_task_assistance(input_prompt, image_data):
    try:
        model = genai.GenerativeModel("gemini-1.5-pro")
        response = model.generate_content([input_prompt, image_data[0]])
        return response.text
    except Exception as e:
        return f"⚠ Error generating task assistance: {str(e)}"

# --- Text-to-Speech ---
engine = pyttsx3.init()

def generate_audio_file(text):
    audio = BytesIO()
    try:
        engine.save_to_file(text, "output.mp3")
        engine.runAndWait()
        with open("output.mp3", "rb") as file:
            audio.write(file.read())
        os.remove("output.mp3")  # Cleanup
        audio.seek(0)  # Reset BytesIO pointer
    except Exception as e:
        raise RuntimeError(f"Text-to-Speech Error: {str(e)}")
    return audio

# --- Translation ---
translator = Translator()

def translate_text(text, target_language):
    if not target_language:
        return text
    try:
        translated = translator.translate(text, dest=target_language)
        return translated.text
    except Exception as e:
        return f"⚠ Translation Error: {str(e)}"

# --- OCR Text Extraction ---
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def extract_text_from_image(image):
    return pytesseract.image_to_string(image)

# --- Text-to-Speech Integration ---
def text_to_speech(text):
    st.write("🔊 Playing Audio...")
    audio = generate_audio_file(text)
    st.audio(audio, format="audio/mp3")

# --- Streamlit Application ---
def main():
    configure_generative_ai("AIzaSyCfxQdg-kdpYMPCTJu9JoLs7koXeld2Vcs")
    object_detection_model = load_object_detection_model()

    st.set_page_config(page_title="AssistVision: Visionary assistance for visually impaired users.", page_icon="👁")

    st.markdown(
        "<h1 style='text-align: center; color: white;'>AssistVision: Visionary assistance for visually impaired users</h1>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<p style='text-align: center;'>Upload an image for detailed analysis and assistance</p>",
        unsafe_allow_html=True
    )
    st.markdown("---")

    feature_choice = st.radio(
        "Choose a feature to interact with:",
        options=["🔍 Describe Scene", "📝 Extract Text", "🚧 Object Detection", "🤖 Personalized Assistance"]
    )

    uploaded_file = st.file_uploader("Upload an Image (JPG, PNG)", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        image_data = [{"mime_type": uploaded_file.type, "data": uploaded_file.getvalue()}]

        if feature_choice == "🔍 Describe Scene":
            with st.spinner("Generating scene description..."):
                scene_prompt = "Describe the scene."
                scene_response = generate_scene_description(scene_prompt, image_data)
                st.write(scene_response)
                if st.button("🔊 Convert Description to Audio"):
                    text_to_speech(scene_response)

        elif feature_choice == "📝 Extract Text":
            with st.spinner("Extracting text using OCR..."):
                extracted_text = extract_text_from_image(image)
                st.write("### Extracted Text:")
                st.text_area("OCR Result", extracted_text, height=200)
                if st.button("🔊 Convert Extracted Text to Audio"):
                    text_to_speech(extracted_text)

        elif feature_choice == "🚧 Object Detection":
            predictions = detect_objects(image, object_detection_model)
            image_with_boxes = draw_boxes(image.copy(), predictions)
            st.image(image_with_boxes, caption="Detected Objects")

        elif feature_choice == "🤖 Personalized Assistance":
            task_prompt = "Assist with tasks based on image data."
            assistance = generate_task_assistance(task_prompt, image_data)
            st.write(assistance)
            if st.button("🔊 Convert Assistance to Audio"):
                text_to_speech(assistance)

if __name__ == "__main__":
    main()

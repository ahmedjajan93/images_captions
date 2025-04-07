import requests
from PIL import Image
from io import BytesIO, StringIO
from bs4 import BeautifulSoup
from transformers import pipeline
import streamlit as st
import torch
import os

st.title("📡 AI Image Captioning from Webpages")

# Initialize session state
if 'processed' not in st.session_state:
    st.session_state.processed = False
    st.session_state.captions = ""

# Load the pretrained processor and model
@st.cache_resource
def load_model():
    model = pipeline(
        "image-to-text",
        model="nlpconnect/vit-gpt2-image-captioning",
        device=-1  
    )
    return model
model = load_model()

# URL input
url = st.text_input("Enter the URL of the page to scrape:", placeholder="https://example.com")

def process_images(img_elements):
    captions = StringIO()
    progress_placeholder = st.empty()
    progress_placeholder.info("🔄 Processing images... This may take a while")
    progress_bar = st.progress(0)

    total_images = len(img_elements)
    processed_count = 0

    for i, img_element in enumerate(img_elements):
        progress_bar.progress((i + 1) / total_images)

        img_url = img_element.get('src')
        if not img_url or 'svg' in img_url or '1x1' in img_url:
            continue

        if img_url.startswith('//'):
            img_url = 'https:' + img_url
        elif not img_url.startswith(('http://', 'https://')):
            continue

        try:
            response = requests.get(img_url, stream=True, timeout=10)
            response.raise_for_status()

            raw_image = Image.open(BytesIO(response.content))
            if raw_image.size[0] * raw_image.size[1] < 400:
                continue

            raw_image = raw_image.convert('RGB')

            caption = model(raw_image)

            captions.write(f"{img_url}: {caption}\n\n")
            processed_count += 1

        except Exception as e:
            st.warning(f"⚠️ Error processing image {img_url}: {str(e)}")
            continue

    progress_bar.empty()
    progress_placeholder.empty()

    if processed_count == 0:
        return "⚠️ No suitable images were processed."
    return captions.getvalue()

# Run processing when button is clicked
if url and st.button("Get Captions", type="primary"):
    try:
        with st.spinner("🌐 Fetching webpage content..."):
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            img_elements = soup.find_all('img')

        if not img_elements:
            st.warning("🔍 No images found on this page!")
        else:
            st.session_state.captions = process_images(img_elements)
            st.session_state.processed = True
            st.success("✅ Captioning completed!")

    except Exception as e:
        st.error(f"❌ Error processing URL: {str(e)}")

# Show download button only after processing
if st.session_state.processed and st.session_state.captions:
    st.download_button(
        label="📥 Download Captions",
        data=st.session_state.captions,
        file_name="captions.txt",
        mime="text/plain"
    )

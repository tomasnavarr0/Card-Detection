import streamlit as st
from models import DetectCards
from envido.cantar_envido import cantar_envido
import cv2
from PIL import Image
import numpy as np

def envido_from_img(img_path: str):
    image_rgb, cards = DetectCards.detect_and_show_cards_in_image(image_path=img_path)
    envido = cantar_envido(cards)
    st.write(envido)
    return envido

def envido_from_video(video_path: str):
    cards = DetectCards.detect_and_show_cards_in_video(video_path=video_path)
    envido = cantar_envido(cards)
    st.write(envido)
    return envido

def envido_from_stream():
    cards = DetectCards.detect_and_show_cards_real_time()
    envido = cantar_envido(cards)
    st.write(envido)
    return envido

def display_image_with_boxes(img_path):
    # Cargar la imagen con OpenCV y realizar la detección
    image_rgb, cards = DetectCards.detect_and_show_cards_in_image(img_path)
    
    # Convertir la imagen a formato compatible con Streamlit
    pil_img = Image.fromarray(image_rgb)
    
    return pil_img, cards

# Streamlit Interface
st.title("Truco Copilot")

# Mostrar la imagen de 1 de espada
#st.image("path/to/1_of_swords_image.jpg", caption="1 de Espada - Naipes Españoles", use_column_width=True)

# Image Upload
st.header("Subir una Imagen")
uploaded_image = st.file_uploader("Elige una imagen", type=["jpg", "jpeg", "png"])
if uploaded_image is not None:
    img_path = f"temp_image.{uploaded_image.type.split('/')[-1]}"
    with open(img_path, "wb") as f:
        f.write(uploaded_image.getbuffer())
    pil_img, cards = display_image_with_boxes(img_path)
    st.image(pil_img, caption="Imagen con Bounding Boxes", use_column_width=True)
    envido_from_img(img_path)

# Video Upload
st.header("Subir un Video")
uploaded_video = st.file_uploader("Elige un video", type=["mp4", "avi", "mov"])
if uploaded_video is not None:
    video_path = f"temp_video.{uploaded_video.type.split('/')[-1]}"
    with open(video_path, "wb") as f:
        f.write(uploaded_video.getbuffer())
    st.video(video_path)
    envido_from_video(video_path)

# Real-time Stream (this part is more complex and might need additional setup)
st.header("Transmisión en Tiempo Real")
if st.button("Iniciar Transmisión en Tiempo Real"):
    envido_from_stream()

import streamlit as st
import cv2
import numpy as np
import json
import os
import pyautogui
from datetime import datetime
import difflib
from PIL import Image
import base64
import io
import requests

# Load configuration
def load_config():
    """Load configuration from file"""
    try:
        with open('config.json', 'r') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading config: {str(e)}")
        return None

def capture_screen(region):
    """Capture a specific region of the screen"""
    screenshot = pyautogui.screenshot(region=(
        region['x'], region['y'], 
        region['width'], region['height']
    ))
    return cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)

def process_image_with_ollama(image, ollama_url):
    """Send image to Ollama for OCR processing"""
    try:
        # Convert BGR to RGB and then to PNG
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)
        
        # Save to bytes buffer
        img_byte_arr = io.BytesIO()
        pil_image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        
        # Encode to base64
        img_base64 = base64.b64encode(img_byte_arr).decode('utf-8')
        
        # Prepare request
        headers = {'Content-Type': 'application/json'}
        payload = {
            'model': 'gemma3:4b',
            'prompt': 'You are an OCR assistant. Extract and return only the text from this image, exactly as shown, without any additional formatting or commentary.',
            'images': [img_base64],
            'stream': False,
            'options': {
                'temperature': 0.1,
                'num_predict': 100
            }
        }
        
        # Make request
        response = requests.post(
            f"{ollama_url}/api/generate",
            headers=headers,
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            if 'response' in result:
                return result['response'].strip()
        return None
            
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        return None

def save_training_data(config, question_img, answer_imgs, question_text, answer_texts, correct_choice):
    """Save captured images and labels"""
    try:
        # Create timestamp for unique filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Ensure directories exist
        os.makedirs(config['training']['images_dir'], exist_ok=True)
        
        # Save images
        image_paths = {
            'question': f"{config['training']['images_dir']}/question_{timestamp}.png",
            'answers': {}
        }
        
        # Save question image
        cv2.imwrite(image_paths['question'], question_img)
        
        # Save answer images
        for choice in ['A', 'B', 'C', 'D']:
            path = f"{config['training']['images_dir']}/answer_{choice}_{timestamp}.png"
            cv2.imwrite(path, answer_imgs[choice])
            image_paths['answers'][choice] = path
        
        # Load existing labels or create new
        labels_file = config['training']['labels_file']
        if os.path.exists(labels_file):
            with open(labels_file, 'r') as f:
                labels = json.load(f)
        else:
            labels = []
        
        # Add new label entry
        label_entry = {
            'timestamp': timestamp,
            'image_paths': image_paths,
            'question_text': question_text,
            'answer_texts': answer_texts,
            'correct_choice': correct_choice,
            'verified': False  # Flag for manual verification
        }
        
        labels.append(label_entry)
        
        # Save updated labels
        with open(labels_file, 'w') as f:
            json.dump(labels, f, indent=2)
            
        return True
        
    except Exception as e:
        st.error(f"Error saving training data: {str(e)}")
        return False

def main():
    st.title("HPMA Quiz Label Tool")
    
    # Load configuration
    config = load_config()
    if config is None:
        st.error("Failed to load configuration. Please check config.json")
        return
    
    # Initialize session state
    if 'labels_modified' not in st.session_state:
        st.session_state.labels_modified = False
    
    # Sidebar for configuration and stats
    with st.sidebar:
        st.header("Configuration")
        
        # Test connection button
        if st.button("Test Ollama Connection"):
            try:
                response = requests.get(f"{config['ollama_url']}/api/tags")
                if response.status_code == 200:
                    st.success("Connected to Ollama server!")
                else:
                    st.error("Failed to connect to Ollama server")
            except Exception as e:
                st.error(f"Connection error: {str(e)}")
        
        # Show current stats
        st.header("Statistics")
        labels_file = config['training']['labels_file']
        if os.path.exists(labels_file):
            with open(labels_file, 'r') as f:
                labels = json.load(f)
                st.write(f"Total samples: {len(labels)}")
                verified = sum(1 for label in labels if label.get('verified', False))
                st.write(f"Verified samples: {verified}")
    
    # Main content area - split into two columns
    left_col, right_col = st.columns([3, 2])
    
    # Left column - Capture and Preview
    with left_col:
        if st.button("📸 Capture Images", use_container_width=True):
            with st.spinner("Capturing and processing..."):
                # Capture images
                question_img = capture_screen(config['question_region'])
                answer_imgs = {}
                for choice in ['A', 'B', 'C', 'D']:
                    answer_imgs[choice] = capture_screen(config['answer_regions'][choice])
                
                # Process with OCR
                question_text = process_image_with_ollama(question_img, config['ollama_url'])
                answer_texts = {}
                for choice in ['A', 'B', 'C', 'D']:
                    answer_texts[choice] = process_image_with_ollama(answer_imgs[choice], config['ollama_url'])
                
                # Store in session state
                st.session_state.question_img = question_img
                st.session_state.answer_imgs = answer_imgs
                st.session_state.question_text = question_text
                st.session_state.answer_texts = answer_texts
        
        # Display captured images
        if 'question_img' in st.session_state:
            st.image(st.session_state.question_img, caption="Question", use_column_width=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.image(st.session_state.answer_imgs['A'], caption="A", use_column_width=True)
                st.image(st.session_state.answer_imgs['C'], caption="C", use_column_width=True)
            with col2:
                st.image(st.session_state.answer_imgs['B'], caption="B", use_column_width=True)
                st.image(st.session_state.answer_imgs['D'], caption="D", use_column_width=True)
    
    # Right column - Labels and Verification
    with right_col:
        if 'question_text' in st.session_state:
            st.markdown("<div style='font-size: 2.4rem;'><strong>Question Text:</strong></div>", unsafe_allow_html=True)
            question_text = st.text_area("##", 
                                       value=st.session_state.question_text if st.session_state.question_text else "",
                                       height=120)  # Reduced height
            
            st.markdown("<div style='font-size: 2.4rem; margin-top: 0.3rem;'><strong>Answer Texts:</strong></div>", unsafe_allow_html=True)
            answer_texts = {}
            for choice in ['A', 'B', 'C', 'D']:
                st.markdown(f"<div style='font-size: 2rem; margin: 0;'>Answer {choice}:</div>", unsafe_allow_html=True)
                answer_texts[choice] = st.text_input(
                    "##",
                    value=st.session_state.answer_texts[choice] if st.session_state.answer_texts[choice] else "",
                    key=f"answer_{choice}"
                )
            
            st.markdown("<div style='font-size: 2.4rem; margin-top: 0.3rem;'><strong>Correct Answer:</strong></div>", unsafe_allow_html=True)
            correct_choice = st.radio(
                "##",
                ['A', 'B', 'C', 'D'],
                horizontal=True
            )
            
            if st.button("Save Training Data", use_container_width=True):
                if save_training_data(
                    config,
                    st.session_state.question_img,
                    st.session_state.answer_imgs,
                    question_text,
                    answer_texts,
                    correct_choice
                ):
                    st.success("Training data saved successfully!")
                    st.session_state.labels_modified = True

if __name__ == "__main__":
    # Set page config
    st.set_page_config(
        page_title="HPMA Quiz Label Tool",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom CSS for bigger text and compact layout
    st.markdown("""
    <style>
        /* Make text areas and inputs much bigger but compact */
        .stTextArea textarea {
            font-size: 2.4rem !important;
            line-height: 1.4 !important;
            padding: 0.5rem !important;
            min-height: 100px !important;
            margin: 0 !important;
        }
        
        .stTextInput input {
            font-size: 2.4rem !important;
            line-height: 1.4 !important;
            padding: 0.3rem 0.5rem !important;
            margin: 0 !important;
            min-height: 40px !important;
        }
        
        /* Compact labels */
        .stMarkdown p, .stMarkdown strong {
            font-size: 2rem !important;
            font-weight: 600 !important;
            margin: 0.3rem 0 !important;
            color: #ffffff !important;
        }
        
        /* Compact radio buttons */
        .stRadio [role="radiogroup"] {
            font-size: 2rem !important;
            margin: 0.3rem 0 !important;
            padding: 0 !important;
        }
        
        .stRadio label {
            font-size: 2rem !important;
            padding: 0.2rem !important;
            margin: 0 !important;
        }
        
        /* Compact buttons */
        .stButton button {
            font-size: 2rem !important;
            padding: 0.5rem 1rem !important;
            margin: 0.5rem 0 !important;
        }

        /* Remove extra spacing between elements */
        .element-container {
            margin: 0.2rem 0 !important;
        }

        /* Make the layout more compact */
        .block-container {
            padding-top: 1rem !important;
            padding-bottom: 0 !important;
        }

        /* Remove default margins from headers */
        h1, h2, h3 {
            margin: 0 !important;
            padding: 0 !important;
        }
    </style>
    """, unsafe_allow_html=True)

    main() 
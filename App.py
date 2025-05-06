import streamlit as st
import cv2
import numpy as np
import pandas as pd
import requests
import json
import os
from PIL import Image
import pyautogui
import difflib
import base64
import io
from ocr_processor import OCRProcessor

# Configuration file path
CONFIG_FILE = 'config.json'

# Initialize OCR processor
@st.cache_resource
def get_ocr_processor():
    return OCRProcessor()

# Set page config to wide mode and remove padding
st.set_page_config(
    page_title="HPMA Quiz OCR Assistant",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply global styles
st.markdown("""
<style>
    /* Remove padding and margin from the main container */
    .main .block-container {
        padding: 0 !important;
        margin: 0 !important;
        max-width: 100% !important;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Custom styling */
    .big-button {
        height: 100px !important;
        padding: 20px !important;
        font-size: 24px !important;
        margin: 20px 0 !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
    }
    
    .captured-image {
        border: 2px solid #333;
        border-radius: 10px;
        margin-bottom: 10px;
    }
    
    .qa-display {
        background-color: #1E1E1E;
        padding: 20px;
        border-radius: 15px;
        margin-top: 20px;
    }
    
    .question-display {
        background-color: #2A2A2A;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        font-size: 18px;
    }
    
    .answer-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 15px;
    }
    
    .answer-box {
        background-color: #2A2A2A;
        padding: 15px 20px;
        border-radius: 10px;
        font-size: 16px;
    }
    
    .correct-answer {
        background-color: #2d5a27;
        border: 2px solid #4CAF50;
    }
    
    .config-section {
        background-color: #1E1E1E;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 15px;
    }
    
    .status-box {
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 10px;
        text-align: center;
    }
    
    .success-box {
        background-color: #2d5a27;
        border: 1px solid #4CAF50;
    }
    
    .preview-container {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 10px;
        margin: 10px 0;
    }
    
    /* Styles for correct answer */
    .correct-answer-header {
        color: #4CAF50;
        font-size: 36px !important;
        font-weight: bold;
        margin-top: 20px;
        margin-bottom: 10px;
    }
    
    .correct-answer-box {
        background-color: #1E3320;
        border: 2px solid #4CAF50;
        border-radius: 10px;
        padding: 15px 20px;
        font-size: 36px !important;
        color: #4CAF50;
        margin-top: 5px;
    }
    
    /* New styles for not found/unmatched cases */
    .not-found-header {
        color: #ff4444;
        font-size: 36px !important;
        font-weight: bold;
        margin-top: 20px;
        margin-bottom: 10px;
    }
    
    .not-found-box {
        background-color: #331E1E;
        border: 2px solid #ff4444;
        border-radius: 10px;
        padding: 15px 20px;
        font-size: 36px !important;
        color: #ff4444;
        margin-top: 5px;
    }
</style>
""", unsafe_allow_html=True)

def migrate_config(old_config):
    """Migrate old config format to new format"""
    new_config = {
        'question_region': old_config['question_region'],
        'answer_regions': {
            'A': {'x': old_config['answer_region']['x'], 
                  'y': old_config['answer_region']['y'], 
                  'width': 300, 'height': 100},
            'B': {'x': old_config['answer_region']['x'] + 320, 
                  'y': old_config['answer_region']['y'], 
                  'width': 300, 'height': 100},
            'C': {'x': old_config['answer_region']['x'], 
                  'y': old_config['answer_region']['y'] + 120, 
                  'width': 300, 'height': 100},
            'D': {'x': old_config['answer_region']['x'] + 320, 
                  'y': old_config['answer_region']['y'] + 120, 
                  'width': 300, 'height': 100}
        },
        'ollama_url': old_config['ollama_url']
    }
    return new_config

def load_config():
    """Load configuration from file or create default if not exists"""
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as f:
            config = json.load(f)
            # Check if config needs migration
            if 'answer_region' in config:
                config = migrate_config(config)
                # Save migrated config
                with open(CONFIG_FILE, 'w') as f:
                    json.dump(config, f)
            return config
    else:
        default_config = {
            'question_region': {'x': 100, 'y': 680, 'width': 680, 'height': 190},
            'answer_regions': {
                'A': {'x': 200, 'y': 880, 'width': 300, 'height': 100},
                'B': {'x': 520, 'y': 880, 'width': 300, 'height': 100},
                'C': {'x': 200, 'y': 1000, 'width': 300, 'height': 100},
                'D': {'x': 520, 'y': 1000, 'width': 300, 'height': 100}
            },
            'ollama_url': 'http://172.24.205.62:11434'
        }
        with open(CONFIG_FILE, 'w') as f:
            json.dump(default_config, f)
        return default_config

def save_config(config):
    """Save configuration to file"""
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f)

def capture_screen(region):
    """Capture a specific region of the screen"""
    screenshot = pyautogui.screenshot(region=(
        region['x'], region['y'], 
        region['width'], region['height']
    ))
    return cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)

def process_image_with_ollama(image, ollama_url):
    """Process image with Ollama API"""
    try:
        # Convert image to base64
        _, buffer = cv2.imencode('.png', image)
        image_base64 = base64.b64encode(buffer).decode('utf-8')

        # Prepare the prompt
        prompt = "You are a quiz assistant. Please analyze this image and extract the text content. Focus on identifying the question and answer options. Format your response as JSON with 'question' and 'answers' fields."

        # Prepare the request
        headers = {'Content-Type': 'application/json'}
        data = {
            'model': 'llava',
            'prompt': prompt,
            'stream': False,
            'images': [image_base64]
        }

        # Make the request
        response = requests.post(f"{ollama_url}/api/generate", 
                               headers=headers,
                               json=data,
                               timeout=30)
        
        if response.status_code == 200:
            try:
                # Extract the response text
                result = response.json()
                response_text = result.get('response', '')
                
                # Try to parse JSON from the response
                try:
                    # Find JSON-like content in the response
                    json_start = response_text.find('{')
                    json_end = response_text.rfind('}') + 1
                    if json_start >= 0 and json_end > json_start:
                        json_content = response_text[json_start:json_end]
                        parsed_response = json.loads(json_content)
                        return parsed_response
                except json.JSONDecodeError:
                    st.error("Failed to parse JSON from Ollama response")
                    return None
                
            except Exception as e:
                st.error(f"Error processing Ollama response: {str(e)}")
                return None
        else:
            st.error(f"Ollama API request failed with status code: {response.status_code}")
            return None
            
    except Exception as e:
        st.error(f"Error communicating with Ollama API: {str(e)}")
        return None

def load_questions_data():
    """Load questions and answers from CSV"""
    try:
        return pd.read_csv('HPMA_data.csv', sep='$', names=['question', 'answer'])
    except FileNotFoundError:
        st.error("HPMA_data.csv not found. Please ensure the file exists in the same directory.")
        return None
    except Exception as e:
        st.error(f"Error loading HPMA_data.csv: {str(e)}")
        return None

def get_text_similarity(text1, text2):
    """Get similarity between two texts using multiple methods"""
    if not text1 or not text2:
        return 0
    
    text1 = text1.lower()
    text2 = text2.lower()
    
    # Exact match
    if text1 == text2:
        return 1.0
    
    # Sequence matcher similarity
    seq_similarity = difflib.SequenceMatcher(None, text1, text2).ratio()
    
    # Word-based similarity
    words1 = set(text1.split())
    words2 = set(text2.split())
    word_similarity = len(words1.intersection(words2)) / max(len(words1), len(words2))
    
    # Token-based similarity (handling partial words)
    tokens1 = set(''.join(c for c in text1 if c.isalnum()).lower())
    tokens2 = set(''.join(c for c in text2 if c.isalnum()).lower())
    token_similarity = len(tokens1.intersection(tokens2)) / max(len(tokens1), len(tokens2))
    
    # Return weighted average of similarities
    return max(seq_similarity * 0.5 + word_similarity * 0.3 + token_similarity * 0.2,
              seq_similarity)  # Ensure we don't go below sequence similarity

def find_best_match(text, questions_df):
    """Find the best matching question and its answer"""
    if questions_df is None or text is None:
        return None, None, 0  # Added score return
    
    best_match = None
    best_score = 0
    
    try:
        for _, row in questions_df.iterrows():
            question = row['question']
            score = get_text_similarity(text, question)
            if score > best_score:
                best_score = score
                best_match = row
        
        if best_score >= 0.8:  # Using threshold from guide
            return best_match['question'], best_match['answer'], best_score  # Return score
    except Exception as e:
        st.error(f"Error matching question: {str(e)}")
    return None, None, 0  # Added score return

def get_available_models(ollama_url):
    """Get list of available models from Ollama server"""
    try:
        response = requests.get(f"{ollama_url}/api/tags")
        if response.status_code == 200:
            models = response.json().get('models', [])
            return [model['name'] for model in models]
        return []
    except:
        return []

def main():
    st.title("HPMA Quiz Assistant")
    
    # Load configuration
    if 'config' not in st.session_state:
        st.session_state.config = load_config()

    # Initialize OCR processor
    ocr = get_ocr_processor()
    
    # Initialize session state
    if 'captured_images' not in st.session_state:
        st.session_state.captured_images = {}
    if 'recognized_text' not in st.session_state:
        st.session_state.recognized_text = {}
    if 'best_match' not in st.session_state:
        st.session_state.best_match = None
    
    # Load questions data
    questions_df = load_questions_data()
    
    # Sidebar Configuration
    with st.sidebar:
        st.header("Configuration")
        
        # Test connection button
        if st.button("Test Ollama Connection"):
            try:
                response = requests.get(f"{st.session_state.config['ollama_url']}/api/tags")
                if response.status_code == 200:
                    st.success("Connected to Ollama server!")
                else:
                    st.error("Failed to connect to Ollama server")
            except Exception as e:
                st.error(f"Connection error: {str(e)}")
        
        # Region configuration in expanders
        with st.expander("Question Region"):
            cols = st.columns(4)
            q_x = cols[0].number_input("X", value=st.session_state.config['question_region']['x'], key="q_x")
            q_y = cols[1].number_input("Y", value=st.session_state.config['question_region']['y'], key="q_y")
            q_width = cols[2].number_input("W", value=st.session_state.config['question_region']['width'], key="q_width")
            q_height = cols[3].number_input("H", value=st.session_state.config['question_region']['height'], key="q_height")
        
        with st.expander("Answer Regions"):
            for choice in ['A', 'B', 'C', 'D']:
                st.markdown(f"**Choice {choice}**")
                cols = st.columns(4)
                x = cols[0].number_input(f"X", value=st.session_state.config['answer_regions'][choice]['x'], key=f"a_x_{choice}")
                y = cols[1].number_input(f"Y", value=st.session_state.config['answer_regions'][choice]['y'], key=f"a_y_{choice}")
                width = cols[2].number_input(f"W", value=st.session_state.config['answer_regions'][choice]['width'], key=f"a_width_{choice}")
                height = cols[3].number_input(f"H", value=st.session_state.config['answer_regions'][choice]['height'], key=f"a_height_{choice}")
        
        if st.button("Save Configuration"):
            new_config = {
                'question_region': {'x': int(q_x), 'y': int(q_y), 'width': int(q_width), 'height': int(q_height)},
                'answer_regions': {
                    'A': {'x': int(st.session_state[f"a_x_A"]), 'y': int(st.session_state[f"a_y_A"]), 
                          'width': int(st.session_state[f"a_width_A"]), 'height': int(st.session_state[f"a_height_A"])},
                    'B': {'x': int(st.session_state[f"a_x_B"]), 'y': int(st.session_state[f"a_y_B"]), 
                          'width': int(st.session_state[f"a_width_B"]), 'height': int(st.session_state[f"a_height_B"])},
                    'C': {'x': int(st.session_state[f"a_x_C"]), 'y': int(st.session_state[f"a_y_C"]), 
                          'width': int(st.session_state[f"a_width_C"]), 'height': int(st.session_state[f"a_height_C"])},
                    'D': {'x': int(st.session_state[f"a_x_D"]), 'y': int(st.session_state[f"a_y_D"]), 
                          'width': int(st.session_state[f"a_width_D"]), 'height': int(st.session_state[f"a_height_D"])}
                },
                'ollama_url': st.session_state.config['ollama_url']
            }
            save_config(new_config)
            st.session_state.config = new_config
            st.success("Configuration saved!")

    # Main content area - split into two columns
    left_col, right_col = st.columns([3, 2])

    # Left column - Capture and Preview
    with left_col:
        st.subheader("Capture and Process")
        
        # Single button for capture and process - now with custom class
        st.markdown(
            """
            <style>
            div[data-testid="stHorizontalBlock"] > div:first-child button {
                height: 100px;
                font-size: 24px !important;
                margin: 20px 0;
                display: flex;
                align-items: center;
                justify-content: center;
            }
            </style>
            """,
            unsafe_allow_html=True
        )
        if st.button("📸 Capture and Process Quiz", use_container_width=True):
            with st.spinner("Capturing and processing..."):
                # Capture question
                question_img = capture_screen(st.session_state.config['question_region'])
                st.session_state.captured_images['question'] = question_img
                
                # Capture answers
                answer_images = {}
                for option, region in st.session_state.config['answer_regions'].items():
                    answer_images[option] = capture_screen(region)
                st.session_state.captured_images.update(answer_images)
                
                # Process all regions with OCR
                st.session_state.recognized_text = ocr.process_quiz_regions(st.session_state.captured_images)
                
                # Find best match in questions database
                if 'question' in st.session_state.recognized_text:
                    question_text = st.session_state.recognized_text['question']
                    match_result = find_best_match(question_text, questions_df)
                    if match_result:
                        question, answer, similarity = match_result
                        st.session_state.best_match = {
                            'Question': question,
                            'Answer': answer,
                            'similarity': similarity
                        }

        # Display captured images in a compact way
        if st.session_state.captured_images:
            # Show question image
            if 'question' in st.session_state.captured_images:
                st.image(st.session_state.captured_images['question'], caption="Question", use_column_width=True)
            
            # Show answer images in 2x2 grid with minimal height
            if all(option in st.session_state.captured_images for option in ['A', 'B', 'C', 'D']):
                col1, col2 = st.columns(2)
                with col1:
                    st.image(st.session_state.captured_images['A'], caption="A", use_column_width=True)
                    st.image(st.session_state.captured_images['C'], caption="C", use_column_width=True)
                with col2:
                    st.image(st.session_state.captured_images['B'], caption="B", use_column_width=True)
                    st.image(st.session_state.captured_images['D'], caption="D", use_column_width=True)

    # Right column - Display processed text
    with right_col:
        st.subheader("Results")
        
        # Display recognized text
        if st.session_state.recognized_text:
            st.write("Recognized Text:")
            for region, text in st.session_state.recognized_text.items():
                st.text_area(f"{region} text:", text, height=100)
        
        # Display best match if found
        if st.session_state.best_match is not None:
            st.markdown("### Best Match from Database")
            st.markdown(f"**Question:** {st.session_state.best_match['Question']}")
            
            # Find correct choice by comparing answers
            correct_choice = None
            highest_similarity = 0
            
            # Only proceed with matching if we have a valid answer
            if st.session_state.best_match.get('Answer'):
                correct_answer = st.session_state.best_match['Answer']
                
                # Try to find exact match first
                for choice in ['A', 'B', 'C', 'D']:
                    if choice in st.session_state.recognized_text:
                        answer_text = st.session_state.recognized_text[choice]
                        if answer_text and correct_answer:  # Check both are not None
                            if answer_text.lower() == correct_answer.lower():
                                correct_choice = choice
                                highest_similarity = 1.0
                                break
                
                # If no exact match, use similarity matching
                if not correct_choice:
                    for choice in ['A', 'B', 'C', 'D']:
                        if choice in st.session_state.recognized_text:
                            answer_text = st.session_state.recognized_text[choice]
                            if answer_text and correct_answer:  # Check both are not None
                                similarity = get_text_similarity(answer_text, correct_answer)
                                if similarity > highest_similarity and similarity >= 0.8:
                                    highest_similarity = similarity
                                    correct_choice = choice
            
            # Display answer with appropriate styling based on match status
            if correct_choice and st.session_state.best_match.get('Answer'):
                # Found match - green styling
                st.markdown('<p class="correct-answer-header">✓ Correct Answer:</p>', unsafe_allow_html=True)
                st.markdown(
                    f'<div class="correct-answer-box">{correct_choice}. {st.session_state.best_match["Answer"]}</div>',
                    unsafe_allow_html=True
                )
            else:
                # No match - red styling
                if st.session_state.best_match.get('Answer'):
                    st.markdown('<p class="not-found-header">❌ Answer Found But No Match:</p>', unsafe_allow_html=True)
                    st.markdown(
                        f'<div class="not-found-box">{st.session_state.best_match["Answer"]}</div>',
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown('<p class="not-found-header">❌ No Answer Found:</p>', unsafe_allow_html=True)
                    st.markdown(
                        '<div class="not-found-box">No answer found in database</div>',
                        unsafe_allow_html=True
                    )
            
            # Display similarity score if available
            if 'similarity' in st.session_state.best_match:
                st.markdown(f"**Similarity Score:** {st.session_state.best_match['similarity']:.2%}")

if __name__ == "__main__":
    main() 
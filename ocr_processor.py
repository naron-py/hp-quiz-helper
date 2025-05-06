import cv2
import numpy as np
import easyocr
import torch
import re
from pathlib import Path
import logging
from typing import Dict, List, Optional, Union
from PIL import Image

class OCRProcessor:
    def __init__(self):
        """Initialize the OCR processor with EasyOCR"""
        self.logger = logging.getLogger(__name__)
        self.reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
        self.debug_dir = Path('debug_images')
        self.debug_dir.mkdir(exist_ok=True)
        
    def preprocess_image(self, image: Union[str, np.ndarray, Image.Image]) -> np.ndarray:
        """Enhanced image preprocessing for better text recognition"""
        try:
            # Convert input to numpy array
            if isinstance(image, str):
                image = Image.open(image).convert('RGB')
                image_np = np.array(image)
            elif isinstance(image, Image.Image):
                image_np = np.array(image)
            else:
                image_np = image

            # Convert to grayscale
            gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR if len(image_np.shape) == 3 else cv2.COLOR_GRAY2BGR)
            gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
            
            # Denoise
            denoised = cv2.fastNlMeansDenoising(gray, h=10)
            
            # Increase contrast using CLAHE
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(denoised)
            
            # Apply Otsu's thresholding
            _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Convert back to RGB
            processed = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)
            
            return processed
            
        except Exception as e:
            self.logger.error(f"Error preprocessing image: {str(e)}")
            return image_np

    def clean_text(self, text: str) -> str:
        """Clean and normalize OCR text output"""
        if not text:
            return ""
            
        # Remove unwanted characters while preserving essential punctuation
        text = re.sub(r'[^a-zA-Z0-9\s\-.,!?]', '', text)
        
        # Fix spacing issues
        text = re.sub(r'(?<![\'\s])([a-z])([A-Z])', r'\1 \2', text)  # Add space between lowercase and uppercase
        text = re.sub(r'\s+', ' ', text)  # Fix multiple spaces
        text = re.sub(r'\s*-\s*', '-', text)  # Fix spacing around hyphens
        text = re.sub(r'([.,!?])([^\s])', r'\1 \2', text)  # Fix spacing after punctuation
        
        return text.strip()

    def recognize_text(self, image: Union[str, np.ndarray, Image.Image], min_confidence: float = 0.5) -> str:
        """Perform OCR on the image and return recognized text"""
        try:
            # Preprocess the image
            processed_image = self.preprocess_image(image)
            
            # Perform OCR
            results = self.reader.readtext(processed_image)
            
            # Filter and combine results
            text_parts = []
            for bbox, text, confidence in results:
                if confidence >= min_confidence:
                    text_parts.append(text)
            
            # Join and clean the text
            full_text = ' '.join(text_parts)
            cleaned_text = self.clean_text(full_text)
            
            return cleaned_text
            
        except Exception as e:
            self.logger.error(f"Error in OCR process: {str(e)}")
            return ""

    def process_quiz_regions(self, regions: Dict[str, np.ndarray]) -> Dict[str, str]:
        """Process multiple quiz regions (question and answers) and return recognized text"""
        results = {}
        
        for region_name, image in regions.items():
            # Adjust confidence threshold based on region type
            min_confidence = 0.4 if region_name == 'question' else 0.5
            
            # Process the region
            text = self.recognize_text(image, min_confidence)
            results[region_name] = text
            
        return results 
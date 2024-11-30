# AssistVision: AI-Powered Assistance for Visually Impaired Individuals  

### Project Overview  
**AssistVision** is an AI-powered application designed to assist visually impaired individuals in understanding and interacting with their environment. By leveraging advanced AI techniques such as object detection, OCR, text-to-speech conversion, and generative AI, this application enables users to navigate their surroundings, access visual content audibly, and receive personalized assistance for daily tasks.  

---

## Features  

### 1. **Real-Time Scene Understanding**  
- Generates descriptive textual interpretations of uploaded images.  
- Allows users to comprehend complex scenes effectively.  
- Includes an optional **Text-to-Speech (TTS)** feature to convert descriptions into audible output.  

### 2. **Text-to-Speech Conversion for Visual Content**  
- Extracts text from uploaded images using **OCR (Optical Character Recognition)**.  
- Converts extracted text into audible speech for seamless accessibility.  

### 3. **Object and Obstacle Detection**  
- Identifies objects and obstacles within images using **Faster R-CNN with ResNet50 backbone**.  
- Highlights detected objects with bounding boxes and confidence scores.  

### 4. **Personalized Assistance for Daily Tasks**  
- Provides task-specific guidance by interpreting image content.  
- Examples include reading product labels or identifying objects for everyday tasks.  

---

## Implementation Details  

### **Technology Stack**  
- **Framework**: Streamlit  
- **Machine Learning**: PyTorch, torchvision (Faster R-CNN for object detection)  
- **OCR**: pytesseract  
- **Text-to-Speech**: pyttsx3  
- **Generative AI**: Google Generative AI (`gemini-1.5-pro`)  
- **Translation**: googletrans  

### **Key Libraries**  
- `torch`, `torchvision`  
- `pytesseract` for OCR  
- `pyttsx3` for TTS  
- `google.generativeai` for scene understanding and task assistance  
- `streamlit` for user interface  

---

## Installation and Setup  

### **Pre-requisites**  
- Python 3.8 or later  
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) installed and configured.  

### **Installation Steps**  
1. Clone the repository:  
   ```bash
   git clone https://github.com/your-repo-name/assistvision.git  
   cd assistvision  
   ```  

2. Install dependencies:  
   ```bash
   pip install -r requirements.txt  
   ```  

3. Configure Tesseract OCR:  
   Ensure the Tesseract binary is accessible. Update the path in the script if necessary:  
   ```python
   pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"  
   ```  

4. Set up Google Generative AI API:  
   Replace the placeholder API key in `configure_generative_ai()` with your key:  
   ```python
   configure_generative_ai("YOUR_GOOGLE_API_KEY")  
   ```  

5. Run the application:  
   ```bash
   streamlit run app.py  
   ```  

---

## Usage  

1. Launch the application by visiting the local Streamlit URL displayed in the terminal.  
2. Choose one of the features:  
   - **Describe Scene**: Upload an image to generate a scene description.  
   - **Extract Text**: Perform OCR on the uploaded image and extract text.  
   - **Object Detection**: Identify objects in the uploaded image.  
   - **Personalized Assistance**: Get task-specific guidance based on the uploaded image.  
3. View the output on the screen. Use the **audio options** to convert text output into speech if required.  

---

## Evaluation Criteria  

### **Functional Completeness**  
- Implementation of at least two assistive features.  

### **Technical Accuracy**  
- Use of state-of-the-art models and techniques for object detection, OCR, and generative AI.  

### **User Experience**  
- Intuitive and accessible UI for visually impaired users.  

### **Documentation**  
- Clear instructions for setup, usage, and customization.  

---

## Future Enhancements  

- **Real-Time Camera Integration**: Extend functionality to process real-time video streams.  
- **Language Support**: Enable support for multiple languages in TTS and OCR.  
- **Edge Deployment**: Optimize for deployment on edge devices like Raspberry Pi.  
- **Accessibility Features**: Include voice-activated commands for hands-free interaction.  

---

## Acknowledgments  

- **Google Generative AI** for enabling powerful scene understanding capabilities.  
- **Tesseract OCR** for reliable text extraction.  
- **PyTorch and torchvision** for object detection models.  
- **Streamlit** for building an accessible and user-friendly interface.  

---

**Developed by**: VENKATA GUNASEKHAR GORRE  
  

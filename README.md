
# AI Chatbot and Image Generation App

This project is an AI-powered application that processes both audio and text inputs, performs Tamil transcription and translation, and responds either via a chatbot or by generating images based on the user’s input. The app integrates several technologies, including Stable Diffusion XL for image generation, Groq API for chat and transcription, and Google Translator for Tamil-to-English translation.

## Features
- Audio Transcription: Upload an audio file in Tamil for transcription using Whisper models.
- Text Translation: Translate the transcribed or inputted Tamil text to English using Google Translator.
- Chatbot Interaction: Get responses from a chatbot powered by the llama-3.2-90b-text-preview model when the input doesn't request image generation.
- Image Generation: Generate images from text prompts using the Stable Diffusion XL model for certain queries that mention image generation (e.g., "create image", "generate picture").
- User Interface: An interactive Gradio-based interface that allows users to input either text or audio, and get outputs like transcription, chatbot response, translations, or generated images.

## How It Works
### 1. Input Methods:
- Text Input: Enter Tamil text directly in the textbox.
- Audio Input: Upload an audio file to transcribe Tamil speech.
### 2. Processing:
- If audio is uploaded, the app transcribes it into Tamil text using Groq's Whisper model.
- The Tamil text is then translated into English using Google Translator.
- The app checks whether the translated input relates to image generation (by scanning for keywords like "image", and "picture").
- The app uses Stable Diffusion XL to generate the image if image generation is detected.
- If not, the chatbot responds with a relevant message using the llama-3.2-90b-text-preview model.
### Output:
- Tamil Transcription / Chatbot Response: Displays the Tamil transcription or chatbot response.
- English Translation: Shows the translated English text.
- Generated Image: Displays the generated image if the input requests image creation.
## Deployment on AWS EC2
### Prerequisites
- AWS Account: Ensure you have an active AWS account.
- EC2 Instance: Set up an EC2 instance with a CUDA-enabled GPU (for running models). The recommended instance types are g4dn.xlarge or higher, depending on your usage requirements.
- SSH Access: Make sure you can SSH into your EC2 instance.
- Domain (optional): You can set up a domain or access the app using the EC2 public IP.
### EC2 Setup Steps
#### 1. Launch an EC2 Instance:
- Choose a GPU-based instance (like g4dn.xlarge).
- Select an appropriate AMI (e.g., Ubuntu 20.04).
- Configure security group settings to open ports for HTTP (port 80) and Gradio (default port 7860).
#### 2. SSH into Your Instance:
```bash
ssh -i "your-key.pem" ubuntu@ec2-your-public-ip.compute.amazonaws.com
```
#### 3. Install Dependencies:
- Update and install necessary packages:
```bash
sudo apt update
sudo apt install python3-pip git -y
```
- Clone your project repository:
```bash
git clone https://github.com/your-username/ai-chatbot-image-generation.git
cd ai-chatbot-image-generation
```
- Install the required Python libraries:
```bash
pip3 install torch gradio groq deep_translator diffusers huggingface_hub safetensors
```
#### 4. Configure Environment Variables:
- Set up your Groq API key in the script (api_key variable).
- Ensure your Hugging Face token is set up if needed for accessing models.
#### 5. Run the Gradio App:
```bash
python3 app.py
```
#### 6. Access the App:
- Open your browser and go to http://ec2-your-public-ip.compute.amazonaws.com:7860 to access the application.
- 
## Setup Installation in Local computer
### Prerequisites
- Python 3.8+
- CUDA-enabled GPU for running the models.
- Access to Groq API for transcription and chatbot interaction.
- Hugging Face API key for accessing models from Hugging Face Hub.
- 
### 1. Clone this repository:
```bash
  git clone https://github.com/your-username/ai-chatbot-image-generation.git
cd ai-chatbot-image-generation
```
### 2. Install the required Python packages:
```bash
  pip install torch gradio groq deep_translator diffusers huggingface_hub safetensors
```
### 3. Set up your environment variables:
- Replace API_key in the script with your Groq API key.
- Ensure you have a Hugging Face token and access to the stabilityai/stable-diffusion-xl-base-1.0 model.
### 4. Launch the Gradio app:
```bash
    python app.py
```
## Usage
- Open the app in your browser (default: http://localhost:7860)
- Click Submit to get the outputs (transcription, translation, chatbot response, or image generation)
- Click Clear to reset the inputs and outputs.
## Models & APIs Used
- Stable Diffusion XL (stabilityai/stable-diffusion-xl-base-1.0): For generating images from text prompts.
- Groq API: This is for audio transcription and chatbot interaction.
    1. Whisper model (whisper-large-v3) for Tamil audio transcription.
    2. Llama model (llama-3.2-90b-text-preview) for chatbot responses.
- Google Translator: This is for translating Tamil to English.
## Acknowledgments
- Stable Diffusion XL for image generation.
- Groq for the audio transcription and chatbot API.
- Google Translator for translation services.
## Contact
For any questions or inquiries, feel free to contact me at infogramrk@gmail.com.

import torch
import gradio as gr
from groq import Groq
import os
from deep_translator import GoogleTranslator
from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel, EulerDiscreteScheduler
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

base = "stabilityai/stable-diffusion-xl-base-1.0"
repo = "ByteDance/SDXL-Lightning"
ckpt = "sdxl_lightning_4step_unet.safetensors" # Use the correct ckpt for your step setting!

# Load model.
unet = UNet2DConditionModel.from_config(base, subfolder="unet").to("cuda", torch.float16)
unet.load_state_dict(load_file(hf_hub_download(repo, ckpt), device="cuda"))
pipe = StableDiffusionXLPipeline.from_pretrained(base, unet=unet, torch_dtype=torch.float16, variant="fp16").to("cuda")

# Ensure sampler uses "trailing" timesteps.
pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")

# Replace with your actual API key
api_key = "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"
client = Groq(api_key=api_key)

def process_audio_or_text(input_text, audio_path):
    # If both inputs are None, return None
    if not input_text and not audio_path:
        return None, None, None

    tamil_text, translation, image = None, None, None

    if audio_path:  # Prefer audio input
        try:
            with open(audio_path, "rb") as file:
                transcription = client.audio.transcriptions.create(
                    file=(os.path.basename(audio_path), file.read()),
                    model="whisper-large-v3",
                    language="ta",
                    response_format="verbose_json",
                )
            tamil_text = transcription.text
        except Exception as e:
            return f"An error occurred during transcription: {str(e)}", None, None

        try:
            translator = GoogleTranslator(source='ta', target='en')
            translation = translator.translate(tamil_text)
        except Exception as e:
            return tamil_text, f"An error occurred during translation: {str(e)}", None

    elif input_text:  # No audio input, so use text input
        translation = input_text

    # Check if the translated input is related to image generation or chatbot
    try:
        prompt_check = list(translation.split(" "))
        image_response = ['picture',"pictures","images","image","create","generate"]
        response = "No"
        for i in image_response:
            if i in prompt_check:
                response = "Yes"
                prompt_check.remove(i)
                break
            else:
                pass
        translation = " ".join(prompt_check)
        if response == "No":  # If not for image, respond with chatbot
            chat_completion = client.chat.completions.create(
                messages=[
                    {"role": "user", "content": translation}
                ],
                model="llama-3.2-90b-text-preview"
            )
            chatbot_response = chat_completion.choices[0].message.content
            return tamil_text, chatbot_response, None
        else:  # If it's for image generation
            image = pipe(translation,num_inference_steps=4, guidance_scale=0).images[0]
            return tamil_text, translation, image

    except Exception as e:
        return None, f"An error occurred during chatbot interaction: {str(e)}", None

with gr.Blocks() as iface:
    gr.Markdown("# AI Chatbot and Image Generation App")

    with gr.Row():
        with gr.Column(scale=1):  # Left side (Inputs and Buttons)
            gr.Markdown("## Tamil Input & Output")
            user_input = gr.Textbox(label="Enter Tamil text", placeholder="Type your message here...")
            audio_input = gr.Audio(type="filepath", label="Or upload audio (for Image Generation)")

            # Buttons placed in the same column as the inputs
            submit_btn = gr.Button("Submit")
            clear_btn = gr.Button("Clear")

        with gr.Column(scale=1):  # Right side (Outputs)
            gr.Markdown("## Tamil/English/Image Output")
            text_output_1 = gr.Textbox(label="Tamil Transcription / Chatbot Response", interactive=False)
            text_output_2 = gr.Textbox(label="English Translation", interactive=False)
            image_output = gr.Image(label="Generated Image")

    # Connect the buttons to the functions
    submit_btn.click(fn=process_audio_or_text,
                     inputs=[user_input, audio_input],
                     outputs=[text_output_1, text_output_2, image_output])
    clear_btn.click(lambda: (None, None, None),
                    outputs=[user_input, audio_input, text_output_1, text_output_2, image_output])

iface.launch()

import numpy as np
import wave
import pyaudio
import gradio as gr
import edge_tts
import asyncio
import google.generativeai as genai
from textblob import TextBlob  # Import for sentiment analysis
import json

# Read the API key from the config.json file
with open(r'C:\Users\aswat\OneDrive\Documents\Voice\config.json', 'r') as file:
    config = json.load(file)
    api_key = config['api_key']

# Configure genai with the API key
genai.configure(api_key=api_key)

# Now you can use genai functions as needed
print("genai is configured with the API key.")

# Audio settings
FRAMES_PER_BUFFER = 3200
CHANNELS = 1
FORMAT = pyaudio.paInt16
RATE = 16000
SILENCE_THRESHOLD = 100  # Amplitude threshold for silence detection
SILENCE_DURATION = 3  # Duration in seconds to consider as silence

# Initialize PyAudio
p = pyaudio.PyAudio()

# Initialize conversation context
conversation_history = []

# List of phrases that could indicate red flags
RED_FLAG_KEYWORDS = [
    "bad", "hate", "angry", "upset", "frustrated", "stress", "worried",
    "sick", "negative", "problem", "sad", "feel bad", "don't want",
    "no", "never", "can't", "avoid", "give up", "hopeless"
]

async def process_audio():
    # Start recording audio
    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=FRAMES_PER_BUFFER
    )

    print('start recording...')
    
    frames = []
    silent_chunks = 0
    silence_limit = int(RATE / FRAMES_PER_BUFFER * SILENCE_DURATION)

    while True:
        data = stream.read(FRAMES_PER_BUFFER)
        frames.append(data)

        # Convert audio data to numpy array to analyze sound level
        audio_data = np.frombuffer(data, dtype=np.int16)
        amplitude = np.abs(audio_data).mean()

        # Check if the amplitude is below the silence threshold
        if amplitude < SILENCE_THRESHOLD:
            silent_chunks += 1
        else:
            silent_chunks = 0

        # If silence has lasted for the specified duration, stop recording
        if silent_chunks > silence_limit:
            print("Silence detected. Stopping recording.")
            break

    stream.stop_stream()
    stream.close()

    # Save the recorded audio to a file
    output_file = r"C:\Users\aswat\OneDrive\Documents\Voice\outputs\output_test.wav"
    with wave.open(output_file, 'wb') as obj:
        obj.setnchannels(CHANNELS)
        obj.setsampwidth(p.get_sample_size(FORMAT))
        obj.setframerate(RATE)
        obj.writeframes(b"".join(frames))
    
    return output_file

async def transcribe_audio(audio_file):
    """Transcribe the audio file to text using the generative model."""
    myfile = genai.upload_file(audio_file)
    model = genai.GenerativeModel("gemini-1.5-flash")
    
    result = model.generate_content([myfile, "Transcribe what you hear in the audio file"])
    response_text = result.text
    return response_text

async def generate_response(audio_file):
    global conversation_history

    # Transcribe the audio file to text
    transcription = await transcribe_audio(audio_file)

    # Detect red flags in the transcription
    red_flag_detected, red_flag_message, detected_keywords = detect_red_flags(transcription)

    # Process the audio file with your model
    myfile = genai.upload_file(audio_file)

    model = genai.GenerativeModel("gemini-1.5-flash")
    
    # Build the context by combining previous messages
    context = "\n".join(conversation_history[-5:])  # Get the last 5 messages
    prompt = (f"You are an emotional assistant. You help people by listening to them and provide them "
              "with appropriate advice, but mainly you listen to them and lend them an ear. "
              f"Previous messages:\n{context}\n"
              "Give a suitable response based on the audio and maintain a conversational style. "
              "Acknowledge the positive things mentioned by them. "
              "Furthermore, the response can be a follow-up question, or a piece of advice, or anything "
              "that you deem as suitable as an emotional assistant.")

    result = model.generate_content([myfile, prompt])
    response_text = result.text
    conversation_history.append(response_text)  # Add response to history

    # Convert text response to speech and save as audio file
    response_audio_file = r"C:\Users\aswat\OneDrive\Documents\Voice\outputs\response_audio.mp3"
    await text_to_speech(response_text, response_audio_file)

    return response_text, response_audio_file, red_flag_detected, red_flag_message, detected_keywords

def detect_red_flags(transcription):
    """Detect red flags in the transcription based on sentiment analysis and keywords."""
    detected_keywords = []

    # Check for red flag keywords
    for keyword in RED_FLAG_KEYWORDS:
        if keyword in transcription.lower():
            detected_keywords.append(keyword)

    # Analyze sentiment
    sentiment = TextBlob(transcription).sentiment
    if detected_keywords or sentiment.polarity < 0:  # Negative sentiment
        red_flag_message = "Detected negative sentiment."
        if detected_keywords:
            red_flag_message += " Keywords detected: " + ", ".join(detected_keywords)
        return True, red_flag_message, detected_keywords

    return False, "No red flags detected.", detected_keywords

async def text_to_speech(text, output_file):
    """Convert text to speech using edge-tts and save as audio file."""
    communicate = edge_tts.Communicate(text, "en-US-JennyNeural")
    with open(output_file, "wb") as file:
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                file.write(chunk["data"])

async def talk():
    audio_file = await process_audio()  # Record audio
    response, audio_file_response, red_flag_detected, red_flag_message, detected_keywords = await generate_response(audio_file)  # Generate response
    return response, audio_file_response, red_flag_detected, red_flag_message, audio_file, detected_keywords

def gradio_interface():
    with gr.Blocks() as demo:
        gr.Markdown("### Emotional Assistant")
        audio_button = gr.Button("Start Recording")
        user_audio_output = gr.Audio(label="User Audio", type="filepath")
        output_text = gr.Textbox(label="Assistant Response", interactive=False)
        audio_output = gr.Audio(label="Response Audio", type="filepath")
        red_flag_output = gr.Textbox(label="Red Flag Detection", interactive=False)
        detected_keywords_output = gr.Textbox(label="Detected Keywords", interactive=False)

        def start_recording():
            response, audio_file_response, red_flag_detected, red_flag_message, user_audio_file, detected_keywords = asyncio.run(talk())
            if red_flag_detected:
                red_flag_message = f"⚠️ Red Flag Detected: {red_flag_message}"
            else:
                red_flag_message = "No red flags detected."
            return response, audio_file_response, user_audio_file, red_flag_message, ", ".join(detected_keywords)

        audio_button.click(
            fn=start_recording, 
            outputs=[output_text, audio_output, user_audio_output, red_flag_output, detected_keywords_output]
        )

    demo.launch()

if __name__ == "__main__":
    gradio_interface()
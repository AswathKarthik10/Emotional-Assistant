import numpy as np
import wave
import pyaudio
import gradio as gr
import edge_tts
import asyncio
from textblob import TextBlob
import google.generativeai as genai
import json

# Get the API-key from the config json file
with open('config.json', 'r') as file:
    config = json.load(file)
    api_key = config['api_key']

# Configure genai with the API key
genai.configure(api_key=api_key)

# Audio settings
FRAMES_PER_BUFFER = 3200
CHANNELS = 1
FORMAT = pyaudio.paInt16
RATE = 16000
SILENCE_THRESHOLD = 100
SILENCE_DURATION = 2

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

    print('Start recording...')
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
    output_file = r"outputs/user_voice.wav"
    with wave.open(output_file, 'wb') as obj:
        obj.setnchannels(CHANNELS)
        obj.setsampwidth(p.get_sample_size(FORMAT))
        obj.setframerate(RATE)
        obj.writeframes(b"".join(frames))
    
    return output_file

async def transcribe_audio_streaming(audio_file):
    """Streaming transcription placeholder."""
    # This function should be implemented with a real streaming transcription service
    print("Simulating streaming transcription...")
    transcription = "This is a simulated transcription of the recorded audio."
    return transcription

async def generate_response(audio_file):
    global conversation_history

    # Transcribe the audio file to text
    transcription = await transcribe_audio_streaming(audio_file)

    # Detect red flags and generate a response in parallel
    response_text, red_flag_detected, red_flag_message = await handle_response_and_red_flags(transcription, audio_file)

    # Convert text response to speech and save as audio file
    response_audio_file = "outputs/response_audio.mp3"
    await text_to_speech(response_text, response_audio_file)

    return response_text, response_audio_file, red_flag_detected, red_flag_message

async def handle_response_and_red_flags(transcription, audio_file):
    """Generate a response and detect red flags in parallel."""
    red_flag_task = asyncio.create_task(detect_red_flags(transcription))
    response_task = asyncio.create_task(generate_model_response(audio_file, transcription))

    # Await both tasks
    red_flag_detected, red_flag_message = await red_flag_task
    response_text = await response_task

    conversation_history.append(response_text)
    return response_text, red_flag_detected, red_flag_message

async def generate_model_response(audio_file, transcription):
    """Generate a response using the model."""
    myfile = genai.upload_file(audio_file)
    context = "\n".join(conversation_history[-5:])
    prompt = (f"You are an emotional assistant. You help people by listening to them and provide them "
              f"with appropriate advice, but mainly you listen to them and lend them an ear. "
              f"Previous messages:\n{context}\n"
              "Give a suitable response based on the audio and maintain a conversational style. "
              "Acknowledge the positive things mentioned by them. "
              "Furthermore, the response can be a follow-up question, a piece of advice, or anything "
              "that you deem as suitable as an emotional assistant.")

    model = genai.GenerativeModel("gemini-1.5-flash")
    result = model.generate_content([myfile, prompt])
    response_text = result.text
    return response_text

async def detect_red_flags(transcription):
    """Detect red flags in the transcription based on sentiment analysis and keywords."""
    # Check for red flag keywords
    for keyword in RED_FLAG_KEYWORDS:
        if keyword in transcription.lower():
            return True, "Detected negative sentiment related to: " + keyword

    # Analyze sentiment
    sentiment = TextBlob(transcription).sentiment
    if sentiment.polarity < 0:
        return True, "Detected negative sentiment."

    return False, ""

async def text_to_speech(text, output_file):
    """Convert text to speech using edge-tts and save as audio file."""
    communicate = edge_tts.Communicate(text, "en-US-JennyNeural")
    with open(output_file, "wb") as file:
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                file.write(chunk["data"])

async def talk():
    audio_file = await process_audio()
    response, audio_file_response, red_flag_detected, red_flag_message = await generate_response(audio_file)
    return response, audio_file_response, red_flag_detected, red_flag_message

def gradio_interface():
    with gr.Blocks() as demo:
        gr.Markdown("### Emotional Assistant")
        audio_button = gr.Button("Start Recording")
        output_text = gr.Textbox(label="Assistant Response", interactive=False)
        audio_output = gr.Audio(label="Response Audio", type="filepath")
        red_flag_output = gr.Textbox(label="Red Flag Detection", interactive=False)

        def start_recording():
            response, audio_file_response, red_flag_message = asyncio.run(talk())
            if red_flag_message:
                red_flag_message = f"⚠️ Red Flag Detected: {red_flag_message}"
            else:
                red_flag_message = "No red flags detected."
            return response, audio_file_response, red_flag_message

        audio_button.click(fn=start_recording, outputs=[output_text, audio_output, red_flag_output])

    demo.launch()

if __name__ == "__main__":
    gradio_interface()
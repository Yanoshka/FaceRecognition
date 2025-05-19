import os
import pygame
from gtts import gTTS
import time
import logging
import speech_recognition as sr
import threading
from threading import Thread
from openai_api import query_api

# Initialize pygame mixer
pygame.mixer.init()


def play_audio(filename):
    """ Function to play audio file """
    pygame.mixer.music.load(filename)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():  # Wait for the music to play completely
        time.sleep(0.1)  # Small sleep to avoid locking up the CPU
    pygame.mixer.music.unload()


def interrupt_listener(stop_event):
    """ Function to listen for the stop command """
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        while not stop_event.is_set():
            try:
                audio = recognizer.listen(source, timeout=1, phrase_time_limit=2)
                text = recognizer.recognize_google(audio)
                if text.lower() == "enough":
                    print("Stopping response...")
                    pygame.mixer.music.stop()
                    stop_event.set()
            except (sr.UnknownValueError, sr.WaitTimeoutError, sr.RequestError):
                pass  # Ignore errors and continue listening


def speak_text(text_to_say, stop_event):
    """ Convert text to speech and play it """
    try:
        tts = gTTS(text=text_to_say, lang='uk')
        filename = 'temp_output.mp3'
        tts.save(filename)
        play_audio(filename)
        os.remove(filename)
    except Exception as e:
        logging.error(f"Failed to play text: {e}")
    finally:
        stop_event.set()

def recognize_speech_from_mic():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        r.adjust_for_ambient_noise(source)  # Adjust for ambient noise
        print("Listening for speech...")
        audio = r.listen(source)
        try:
            text = r.recognize_google(audio, language='uk-UA').lower()
            print(f"Recognized text: {text}")
            return text
        except sr.UnknownValueError:
            print("Google Speech Recognition could not understand audio")
        except sr.RequestError as e:
            print(f"Could not request results from Google Speech Recognition service; {e}")
        return None

def main():
    while True:
        stop_event = threading.Event()
        input_text = recognize_speech_from_mic()
        if input_text:
            response_text = query_api(input_text)
            print(f"API response: {response_text}")

            # Start the audio thread
            audio_thread = Thread(target=speak_text, args=(response_text, stop_event))
            audio_thread.start()

            # Start the listening thread
            listener_thread = Thread(target=interrupt_listener, args=(stop_event,))
            listener_thread.start()

            # Wait for either thread to finish
            audio_thread.join()
            listener_thread.join()
        else:
            print("Say something again...")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()



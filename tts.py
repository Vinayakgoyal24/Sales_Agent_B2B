# tts.py
import win32com.client

speaker = win32com.client.Dispatch("SAPI.SpVoice")

# Select a specific voice (if available)
target_name = "Microsoft David"
for voice in speaker.GetVoices():
    if target_name in voice.GetDescription():
        speaker.Voice = voice
        break

speaker.Rate   = 0
speaker.Volume = 90
speaker.Speak("Hi, I am your personal AI agent.")

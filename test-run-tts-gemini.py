"""
Thai Conversation TTS Test - Single Audio Generation
This script generates one Thai conversation audio file for testing purposes.

**Prerequisites:**
1. Install: pip install google-genai
2. Set GOOGLE_API_KEY environment variable
3. Run: python thai_conversation_test.py

**Output:** thai_conversation_test.wav
"""

import os
import sys
import wave
from google import genai
from google.genai import types

def wave_file(filename, pcm, channels=1, rate=24000, sample_width=2):
    """
    Save PCM audio data to a WAV file.
    
    Args:
        filename (str): Output filename
        pcm (bytes): PCM audio data
        channels (int): Number of audio channels (default: 1)
        rate (int): Sample rate (default: 24000)
        sample_width (int): Sample width in bytes (default: 2)
    """
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(rate)
        wf.writeframes(pcm)

def generate_thai_conversation(output_filename="thai_conversation_test.wav"):
    """
    Generates a Thai dialect conversation using different voices.
    """
    try:
        # Check if API key is set
        api_key = os.environ.get('GOOGLE_API_KEY')
        if not api_key:
            print("Error: GOOGLE_API_KEY environment variable is not set.")
            print("Please set your Google AI Studio API key:")
            print('$env:GOOGLE_API_KEY="your_api_key_here"')
            return False
        
        client = genai.Client(api_key=api_key)
        
        # Thai conversation prompt with Southern dialect instruction
        prompt = """Please generate text-to-speech for the following Thai dialect conversation between Speaker 1 and Speaker 2:

Speaker 1: สวัสดีสบายดีไหม
Speaker 2: สบายดี คุณกำลังทำอะไรอยู่หรอ

Please use Thai language pronunciation and intonation appropriate for Thai dialect."""

        print("Generating Thai conversation...")
        print("Thai text:")
        print("Speaker 1: สวัสดีสบายดีไหม")
        print("Speaker 2: สบายดี คุณกำลังทำอะไรอยู่หรอ")
        
        response = client.models.generate_content(
            model="gemini-2.5-flash-preview-tts",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_modalities=["AUDIO"],
                speech_config=types.SpeechConfig(
                    multi_speaker_voice_config=types.MultiSpeakerVoiceConfig(
                        speaker_voice_configs=[
                            types.SpeakerVoiceConfig(
                                speaker='Speaker 1',
                                voice_config=types.VoiceConfig(
                                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                        voice_name='Aoede',  # Female voice for Speaker 1
                                    )
                                )
                            ),
                            types.SpeakerVoiceConfig(
                                speaker='Speaker 2',
                                voice_config=types.VoiceConfig(
                                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                        voice_name='Kore',  # Different female voice for Speaker 2
                                    )
                                )
                            ),
                        ]
                    )
                )
            )
        )

        # Extract and save audio data
        if response.candidates and response.candidates[0].content.parts:
            audio_part = response.candidates[0].content.parts[0]
            if hasattr(audio_part, 'inline_data') and audio_part.inline_data:
                data = audio_part.inline_data.data
                wave_file(output_filename, data)
                print(f"✓ Thai conversation saved to: {output_filename}")
                return True
        
        print("✗ No audio data found in response")
        return False
        
    except Exception as e:
        print(f"✗ Error during Thai speech generation: {e}")
        return False

if __name__ == "__main__":
    # Check if API key is set up
    if not os.environ.get('GOOGLE_API_KEY'):
        print("❌ Error: GOOGLE_API_KEY environment variable is not set.")
        print("Please set your Google AI Studio API key:")
        print('$env:GOOGLE_API_KEY="your_api_key_here"')
        print("\nGet your API key from: https://ai.google.dev/")
        sys.exit(1)
    
    print("🇹🇭 Thai Conversation TTS Test")
    print("=" * 40)
    print("Generating single Thai conversation audio file...")
    print()
    
    # Generate Thai conversation
    success = generate_thai_conversation("thai_conversation_test.wav")
    
    if success:
        print()
        print("🎉 Success!")
        print("📁 Generated file: thai_conversation_test.wav")
        print("🔊 Contains Thai conversation between 2 speakers:")
        print("   Speaker 1 (Aoede): สวัสดีสบายดีไหม")
        print("   Speaker 2 (Kore): สบายดี คุณกำลังทำอะไรอยู่หรอ")
    else:
        print()
        print("❌ Failed to generate audio file")
        print("💡 Check your API key and quota status")

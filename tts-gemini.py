def generate_thai_conversation(output_filename="thai_conversation.wav"):
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

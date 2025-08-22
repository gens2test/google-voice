#!/usr/bin/env python3
"""
Real-time Microphone Streaming with Google Cloud Speaker Diarization
Based on Google's official streaming speaker diarization example:
https://cloud.google.com/speech-to-text/docs/transcribe-streaming-audio
and demo from https://cloud.google.com/speech-to-text?hl=en#demo 

This implementation uses all the real-time features mentioned in the Google example:
- Streaming recognition with speaker diarization enabled
- Real-time processing of speaker tags for each word
- Live speaker statistics and color-coded transcription
- Proper chunk size (100ms) as recommended by Google
"""

import os
import pyaudio
import threading
import time
import queue
import tkinter as tk
from tkinter import ttk, scrolledtext
from google.cloud import speech_v1p1beta1 as speech
from google.oauth2 import service_account
from collections import defaultdict

class GoogleStyleSpeakerDiarization:
    def __init__(self):
        # Audio recording parameters (from example)
        self.RATE = 16000
        self.CHUNK = int(self.RATE / 10)  # 100ms chunks as per Google
        self.CHANNELS = 1
        self.FORMAT = pyaudio.paInt16
        
        # Threading and control
        self.is_recording = False
        self.audio_queue = queue.Queue()
        self.pyaudio_instance = None
        self.stream = None
        self.transcription_thread = None
        
        # Speaker tracking and colors
        self.speaker_colors = {
            1: "#FF6B6B",  # Red
            2: "#4ECDC4",  # Teal  
            3: "#45B7D1",  # Blue
            4: "#96CEB4",  # Green
            5: "#FFEAA7",  # Yellow
            6: "#DDA0DD",  # Plum
        }
        self.speaker_stats = defaultdict(int)  # Track word count per speaker
        
        # Google Cloud Speech client
        self.speech_client = None
        
        # GUI setup (must be first for logging)
        self.setup_gui()
        
        # Initialize Google Cloud after GUI is ready
        self.setup_google_cloud()
        
    def setup_google_cloud(self):
        """Initialize Google Cloud Speech client with service account"""
        try:
            service_account_path = "voice-sun-1-67f7efc777f3.json"
            if not os.path.exists(service_account_path):
                raise FileNotFoundError(f"Service account file not found: {service_account_path}")
                
            credentials = service_account.Credentials.from_service_account_file(
                service_account_path,
                scopes=["https://www.googleapis.com/auth/cloud-platform"]
            )
            
            self.speech_client = speech.SpeechClient(credentials=credentials)
            self.log_message("✅ Google Cloud Speech client initialized successfully")
            
        except Exception as e:
            self.log_message(f"❌ Error setting up Google Cloud: {str(e)}")
            
    def setup_gui(self):
        """Create the main GUI interface"""
        self.root = tk.Tk()
        self.root.title("🎤 Google-Style Real-time Speaker Diarization")
        self.root.geometry("1000x850")  # Increased height to show log section
        self.root.configure(bg="#f0f0f0")
        
        # Title
        title_label = tk.Label(
            self.root, 
            text="🎤 Real-time Speech with Google Speaker Diarization", 
            font=("Arial", 16, "bold"),
            bg="#f0f0f0",
            fg="#333"
        )
        title_label.pack(pady=10)
        
        # Control frame
        control_frame = tk.Frame(self.root, bg="#f0f0f0")
        control_frame.pack(pady=10)
        
        # Start/Stop button
        self.toggle_button = tk.Button(
            control_frame,
            text="🎙️ Start Recording",
            font=("Arial", 12, "bold"),
            bg="#4CAF50",
            fg="white",
            padx=20,
            pady=10,
            command=self.toggle_recording
        )
        self.toggle_button.pack(side=tk.LEFT, padx=10)
        
        # Clear button
        clear_button = tk.Button(
            control_frame,
            text="🗑️ Clear",
            font=("Arial", 12),
            bg="#f44336",
            fg="white",
            padx=20,
            pady=10,
            command=self.clear_transcription
        )
        clear_button.pack(side=tk.LEFT, padx=10)
        
        # Language selection
        lang_frame = tk.Frame(control_frame, bg="#f0f0f0")
        lang_frame.pack(side=tk.LEFT, padx=20)
        
        tk.Label(lang_frame, text="Language:", bg="#f0f0f0", font=("Arial", 10)).pack()
        self.language_var = tk.StringVar(value="en-US")
        language_combo = ttk.Combobox(
            lang_frame, 
            textvariable=self.language_var,
            values=["en-US", "th-TH"],
            width=10,
            state="readonly"
        )
        language_combo.pack()
        
        # Speaker count configuration (as in Google example)
        speaker_frame = tk.Frame(control_frame, bg="#f0f0f0")
        speaker_frame.pack(side=tk.LEFT, padx=20)
        
        tk.Label(speaker_frame, text="Expected Speakers:", bg="#f0f0f0", font=("Arial", 10)).pack()
        self.min_speakers_var = tk.StringVar(value="1")
        self.max_speakers_var = tk.StringVar(value="6")
        
        speaker_config_frame = tk.Frame(speaker_frame, bg="#f0f0f0")
        speaker_config_frame.pack()
        
        tk.Label(speaker_config_frame, text="Min:", bg="#f0f0f0", font=("Arial", 9)).pack(side=tk.LEFT)
        min_spinner = tk.Spinbox(speaker_config_frame, textvariable=self.min_speakers_var, 
                                from_=1, to=6, width=3, font=("Arial", 9))
        min_spinner.pack(side=tk.LEFT, padx=2)
        
        tk.Label(speaker_config_frame, text="Max:", bg="#f0f0f0", font=("Arial", 9)).pack(side=tk.LEFT, padx=(5,0))
        max_spinner = tk.Spinbox(speaker_config_frame, textvariable=self.max_speakers_var, 
                                from_=2, to=6, width=3, font=("Arial", 9))
        max_spinner.pack(side=tk.LEFT, padx=2)
        
        # Main content with tabs
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Real-time transcription tab
        realtime_frame = ttk.Frame(notebook)
        notebook.add(realtime_frame, text="🎙️ Live Transcription with Speakers")
        
        # Speaker statistics frame
        stats_frame = tk.Frame(realtime_frame, bg="white", relief="solid", bd=1)
        stats_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.stats_label = tk.Label(
            stats_frame,
            text="👥 Speaker Statistics: Waiting for speech...",
            font=("Arial", 11, "bold"),
            bg="white",
            fg="#333",
            anchor="w"
        )
        self.stats_label.pack(fill=tk.X, padx=10, pady=8)
        
        # Transcription display with speaker colors
        self.transcript_text = scrolledtext.ScrolledText(
            realtime_frame,
            height=18,  # Reduced height to make room for log
            font=("Arial", 11),
            wrap=tk.WORD,
            bg="white",
            fg="#333"
        )
        self.transcript_text.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Log display
        log_label = tk.Label(
            self.root,
            text="📋 System Log:",
            font=("Arial", 10, "bold"),
            bg="#f0f0f0",
            anchor="w"
        )
        log_label.pack(fill=tk.X, padx=20, pady=(5, 2))
        
        self.log_text = scrolledtext.ScrolledText(
            self.root,
            height=8,  # Increased log height for better visibility
            font=("Arial", 9),
            wrap=tk.WORD,
            bg="#f8f8f8",
            fg="#666"
        )
        self.log_text.pack(fill=tk.X, padx=20, pady=(0, 20))
        
    def log_message(self, message):
        """Add message to log display"""
        timestamp = time.strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}\n"
        
        self.log_text.insert(tk.END, log_entry)
        self.log_text.see(tk.END)
        self.root.update_idletasks()
        
    def add_transcription_with_speakers(self, transcript, words_info=None, is_final=False):
        """Add transcription to display with speaker information (following Google's example)"""
        timestamp = time.strftime("%H:%M:%S")
        
        if words_info and is_final:
            # Process speaker information for final results (as in Google's example)
            self.log_message(f"👥 Processing {len(words_info)} words with speaker tags")
            
            formatted_text = f"[{timestamp}] "
            current_speaker = None
            current_text = ""
            
            for word_info in words_info:
                speaker_tag = word_info.speaker_tag
                word = word_info.word
                
                self.log_message(f"  Word: '{word}' -> Speaker {speaker_tag}")
                
                # Update speaker statistics
                self.speaker_stats[speaker_tag] += 1
                
                if current_speaker != speaker_tag:
                    # Speaker changed, add previous speaker's text
                    if current_text:
                        color = self.speaker_colors.get(current_speaker, "#333333")
                        self.transcript_text.insert(tk.END, current_text)
                        # Apply color to the just-inserted text
                        start_index = self.transcript_text.index(f"end-{len(current_text)}c")
                        end_index = self.transcript_text.index("end-1c")
                        self.transcript_text.tag_add(f"speaker_{current_speaker}", start_index, end_index)
                        self.transcript_text.tag_config(f"speaker_{current_speaker}", 
                                                      foreground=color, font=("Arial", 11, "bold"))
                    
                    # Start new speaker text
                    current_speaker = speaker_tag
                    current_text = f"[Speaker {speaker_tag}] {word} "
                else:
                    current_text += f"{word} "
            
            # Add remaining text
            if current_text:
                color = self.speaker_colors.get(current_speaker, "#333333")
                self.transcript_text.insert(tk.END, current_text)
                start_index = self.transcript_text.index(f"end-{len(current_text)}c")
                end_index = self.transcript_text.index("end-1c")
                self.transcript_text.tag_add(f"speaker_{current_speaker}", start_index, end_index)
                self.transcript_text.tag_config(f"speaker_{current_speaker}", 
                                              foreground=color, font=("Arial", 11, "bold"))
            
            self.transcript_text.insert(tk.END, "\n\n")
            
            # Update speaker statistics display
            self.update_speaker_stats()
            
        elif is_final:
            # Final results without speaker info
            text = f"[{timestamp}] {transcript}\n\n"
            self.transcript_text.insert(tk.END, text)
        else:
            # Interim results (as in Google's example)
            interim_text = f"[Interim] {transcript}\n"
            self.transcript_text.insert(tk.END, interim_text)
            self.transcript_text.tag_add("interim", "end-2l", "end-1l")
            self.transcript_text.tag_config("interim", foreground="#888888", font=("Arial", 10, "italic"))
        
        self.transcript_text.see(tk.END)
        self.root.update_idletasks()
        
    def update_speaker_stats(self):
        """Update the speaker statistics display"""
        if not self.speaker_stats:
            self.stats_label.config(text="👥 Speaker Statistics: Waiting for speech...")
            return
            
        total_words = sum(self.speaker_stats.values())
        stats_text = "👥 Speaker Statistics: "
        
        for speaker, word_count in sorted(self.speaker_stats.items()):
            percentage = (word_count / total_words) * 100
            stats_text += f"Speaker {speaker}: {word_count} words ({percentage:.1f}%) | "
            
        stats_text = stats_text.rstrip(" | ")
        self.stats_label.config(text=stats_text)
        
    def clear_transcription(self):
        """Clear the transcription display"""
        self.transcript_text.delete(1.0, tk.END)
        self.speaker_stats.clear()
        self.update_speaker_stats()
        self.log_message("🗑️ Transcription cleared")
        
    def toggle_recording(self):
        """Toggle recording on/off"""
        if not self.is_recording:
            self.start_recording()
        else:
            self.stop_recording()
            
    def start_recording(self):
        """Start recording and transcription"""
        try:
            self.is_recording = True
            self.toggle_button.config(text="🛑 Stop Recording", bg="#f44336")
            
            # Initialize PyAudio
            self.pyaudio_instance = pyaudio.PyAudio()
            self.stream = self.pyaudio_instance.open(
                format=self.FORMAT,
                channels=self.CHANNELS,
                rate=self.RATE,
                input=True,
                frames_per_buffer=self.CHUNK
            )
            
            self.log_message("🎙️ Microphone recording started")
            
            # Start transcription thread (following Google's pattern)
            self.transcription_thread = threading.Thread(target=self.main_streaming_loop)
            self.transcription_thread.daemon = True
            self.transcription_thread.start()
            
            # Start audio capture thread
            audio_thread = threading.Thread(target=self.audio_capture_worker)
            audio_thread.daemon = True
            audio_thread.start()
            
        except Exception as e:
            self.log_message(f"❌ Error starting recording: {str(e)}")
            self.stop_recording()
            
    def stop_recording(self):
        """Stop recording and transcription"""
        self.is_recording = False
        self.toggle_button.config(text="🎙️ Start Recording", bg="#4CAF50")
        
        try:
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
                self.stream = None
                
            if self.pyaudio_instance:
                self.pyaudio_instance.terminate()
                self.pyaudio_instance = None
                
            self.log_message("🛑 Recording stopped")
            
        except Exception as e:
            self.log_message(f"❌ Error stopping recording: {str(e)}")
            
    def audio_capture_worker(self):
        """Worker thread to capture audio from microphone"""
        self.log_message("🎵 Audio capture thread started")
        
        while self.is_recording:
            try:
                if self.stream:
                    data = self.stream.read(self.CHUNK, exception_on_overflow=False)
                    self.audio_queue.put(data)
            except Exception as e:
                if self.is_recording:
                    self.log_message(f"❌ Audio capture error: {str(e)}")
                break
                
        self.log_message("🎵 Audio capture thread stopped")
        
    def microphone_stream(self):
        """Generates a stream of audio data from the microphone (Google's pattern)"""
        while self.is_recording:
            try:
                chunk = self.audio_queue.get(timeout=0.5)
                yield chunk
            except queue.Empty:
                continue
                
    def main_streaming_loop(self):
        """Main streaming recognition loop (following Google's example exactly)"""
        self.log_message("🚀 Starting streaming recognition with speaker diarization...")
        
        # Get configuration
        current_language = self.language_var.get()
        min_speakers = int(self.min_speakers_var.get())
        max_speakers = int(self.max_speakers_var.get())
        
        self.log_message(f"🌐 Language: {current_language}")
        self.log_message(f"👥 Expected speakers: {min_speakers}-{max_speakers}")
        
        # Configure speaker diarization (from example)
        diarization_config = speech.SpeakerDiarizationConfig(
            enable_speaker_diarization=True,
            min_speaker_count=min_speakers,
            max_speaker_count=max_speakers,
        )
        
        # Recognition configuration (from example)
        recognition_config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=self.RATE,
            language_code=current_language,
            diarization_config=diarization_config,
        )
        
        # Streaming configuration (from example)
        streaming_config = speech.StreamingRecognitionConfig(
            config=recognition_config,
            interim_results=True
        )
        
        # Create audio generator and requests (from example)
        audio_generator = self.microphone_stream()
        requests = (speech.StreamingRecognizeRequest(audio_content=content)
                  for content in audio_generator)
        
        try:
            # Perform streaming recognition (from example)
            responses = self.speech_client.streaming_recognize(streaming_config, requests)
            
            self.log_message("✅ Streaming recognition connected, listening...")
            
            # Process responses (following guide pattern)
            for response in responses:
                if not self.is_recording:
                    break
                    
                if not response.results:
                    continue
                    
                result = response.results[0]
                if not result.alternatives:
                    continue
                    
                transcript = result.alternatives[0].transcript
                
                if result.is_final:
                    self.log_message(f"✅ Final transcript: {transcript}")
                    
                    # The final result with speaker tags (following Google's documentation)
                    words_info = result.alternatives[0].words
                    if words_info:
                        self.log_message(f"👥 Got {len(words_info)} words with speaker information")
                        for word_info in words_info:
                            self.log_message(f"  Word: {word_info.word}, Speaker Tag: {word_info.speaker_tag}")
                    
                    self.add_transcription_with_speakers(transcript, words_info, is_final=True)
                else:
                    # Interim results (following Google's documentation)
                    self.add_transcription_with_speakers(transcript, is_final=False)
                    
        except Exception as e:
            if self.is_recording:
                self.log_message(f"❌ Streaming error: {str(e)}")
                self.stop_recording()
                
    def run(self):
        """Start the application"""
        self.log_message("🚀 Google-Style Real-time Speaker Diarization Started")
        self.log_message("📌 This implementation follows Google's official streaming example")
        self.log_message("💡 Click 'Start Recording' to begin real-time speaker identification")
        self.root.mainloop()

def main():
    """Main function (following Google's documentation)"""
    print("Listening... Press Ctrl+C to stop.")
    app = GoogleStyleSpeakerDiarization()
    app.run()

if __name__ == "__main__":
    main()

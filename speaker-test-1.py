#!/usr/bin/env python3
"""
Real-time Microphone Streaming Transcription with Speaker Diarization
Hybrid approach: Real-time transcription + Batch speaker identification
"""

import os
import sys
import threading
import time
import queue
import wave
import tempfile
import pyaudio
import tkinter as tk
from tkinter import ttk, scrolledtext
from google.cloud import speech
from google.oauth2 import service_account
from collections import deque
import requests                     # For Beta API calls
import json
import base64
from google.auth.transport.requests import Request

class MicrophoneStreamingWithSpeakers:
    def __init__(self):
        # Audio configuration
        self.RATE = 16000
        self.CHUNK = 1024
        self.CHANNELS = 1
        self.FORMAT = pyaudio.paInt16
        
        # Threading and control
        self.is_recording = False
        self.audio_queue = queue.Queue()
        self.audio_buffer = deque()  # Store audio for speaker diarization
        self.buffer_lock = threading.Lock()  # Lock for thread-safe buffer access
        self.pyaudio_instance = None
        self.stream = None
        self.transcription_thread = None
        self.speaker_analysis_thread = None
        
        # Speaker diarization settings
        self.buffer_duration = 10  # seconds
        self.last_speaker_analysis = 0
        self.speaker_segments = []
        
        # Google Cloud Speech client
        self.speech_client = None
        
        # GUI setup (must be before speech client setup for logging)
        self.setup_gui()
        
        # Initialize speech client after GUI is ready
        self.setup_speech_client()
        
    def get_beta_api_credentials(self):
        """Get credentials for beta API authentication"""
        try:
            service_account_path = "voice-sun-1-67f7efc777f3.json"
            if os.path.exists(service_account_path):
                credentials = service_account.Credentials.from_service_account_file(
                    service_account_path,
                    scopes=['https://www.googleapis.com/auth/cloud-platform']
                )
                credentials.refresh(Request())
                return credentials.token
            else:
                self.log_message("❌ Service account file not found!")
                return None
        except Exception as e:
            self.log_message(f"❌ Error getting beta API credentials: {e}")
            return None
        
    def setup_speech_client(self):
        """Initialize Google Cloud Speech client with service account authentication"""
        try:
            # Use service account key file
            service_account_path = "voice-sun-1-67f7efc777f3.json"
            if os.path.exists(service_account_path):
                credentials = service_account.Credentials.from_service_account_file(service_account_path)
                self.speech_client = speech.SpeechClient(credentials=credentials)
                self.log_message("✅ Google Cloud Speech client initialized successfully!")
            else:
                self.log_message("❌ Service account file not found!")
                return False
        except Exception as e:
            self.log_message(f"❌ Error initializing Speech client: {e}")
            return False
        return True
    
    def setup_gui(self):
        """Create the GUI interface"""
        self.root = tk.Tk()
        self.root.title("🎤 Real-time Transcription with Speaker Identification")
        self.root.geometry("900x700")
        self.root.configure(bg="#f0f0f0")
        
        # Title
        title_label = tk.Label(
            self.root, 
            text="🎤 Real-time Speech with Enhanced Speaker ID (Beta API)", 
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
        self.language_var = tk.StringVar(value="th-TH")  # Default to Thai
        language_combo = ttk.Combobox(
            lang_frame, 
            textvariable=self.language_var,
            values=["th-TH", "en-US"],
            width=10,
            state="readonly"
        )
        language_combo.pack()
        
        # Speaker analysis settings
        speaker_frame = tk.Frame(control_frame, bg="#f0f0f0")
        speaker_frame.pack(side=tk.LEFT, padx=20)
        
        tk.Label(speaker_frame, text="Speaker Analysis:", bg="#f0f0f0", font=("Arial", 10)).pack()
        self.speaker_interval_var = tk.StringVar(value="10")
        speaker_combo = ttk.Combobox(
            speaker_frame,
            textvariable=self.speaker_interval_var,
            values=["5", "10", "15", "20"],
            width=8,
            state="readonly"
        )
        speaker_combo.pack()
        tk.Label(speaker_frame, text="seconds", bg="#f0f0f0", font=("Arial", 8)).pack()
        
        # Status frame
        status_frame = tk.Frame(self.root, bg="#f0f0f0")
        status_frame.pack(fill=tk.X, padx=20, pady=5)
        
        self.status_label = tk.Label(
            status_frame,
            text="🔴 Ready to record",
            font=("Arial", 10),
            bg="#f0f0f0",
            fg="#666"
        )
        self.status_label.pack(side=tk.LEFT)
        
        self.speaker_status_label = tk.Label(
            status_frame,
            text="👥 Speaker analysis: Ready",
            font=("Arial", 10),
            bg="#f0f0f0",
            fg="#666"
        )
        self.speaker_status_label.pack(side=tk.RIGHT)
        
        # Create notebook for tabs
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Real-time transcription tab
        realtime_frame = ttk.Frame(notebook)
        notebook.add(realtime_frame, text="📝 Real-time Transcription")
        
        self.transcript_text = scrolledtext.ScrolledText(
            realtime_frame,
            height=15,
            font=("Arial", 11),
            wrap=tk.WORD,
            bg="white",
            fg="#333"
        )
        self.transcript_text.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Speaker analysis tab
        speaker_frame = ttk.Frame(notebook)
        notebook.add(speaker_frame, text="👥 Speaker Analysis")
        
        self.speaker_text = scrolledtext.ScrolledText(
            speaker_frame,
            height=20,
            font=("Arial", 11),
            wrap=tk.WORD,
            bg="#f8f8f8",
            fg="#333"
        )
        self.speaker_text.pack(fill=tk.BOTH, expand=True, pady=10)
        
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
            height=6,
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
        print(log_entry.strip())  # Also print to console
        
    def add_transcription(self, text, is_final=False):
        """Add transcription to real-time display"""
        # Debug: Log what we're adding
        self.log_message(f"🔍 Adding transcription: '{text}' (final: {is_final})")
        
        if is_final:
            # Remove any previous interim result
            current_content = self.transcript_text.get("end-2l", "end-1l").strip()
            if current_content.startswith("🔄"):
                # Delete the interim result line
                self.transcript_text.delete("end-2l", "end-1l")
            
            # Add final result
            timestamp = time.strftime("%H:%M:%S")
            self.transcript_text.insert(tk.END, f"[{timestamp}] {text}\n")
            self.transcript_text.insert(tk.END, "-" * 50 + "\n")
        else:
            # For interim results, check if we need to replace the last interim line
            current_content = self.transcript_text.get("end-2l", "end-1l").strip()
            if current_content.startswith("🔄"):
                # Replace the previous interim result
                self.transcript_text.delete("end-2l", "end-1l")
            
            # Add new interim result
            self.transcript_text.insert(tk.END, f"🔄 {text}\n")
        
        self.transcript_text.see(tk.END)
        self.root.update_idletasks()
        
    def add_speaker_analysis(self, analysis_text):
        """Add speaker analysis to speaker tab"""
        timestamp = time.strftime("%H:%M:%S")
        self.speaker_text.insert(tk.END, f"[{timestamp}] {analysis_text}\n")
        self.speaker_text.insert(tk.END, "-" * 60 + "\n")
        self.speaker_text.see(tk.END)
        self.root.update_idletasks()
        
    def clear_transcription(self):
        """Clear both transcription displays"""
        self.transcript_text.delete(1.0, tk.END)
        self.speaker_text.delete(1.0, tk.END)
        self.speaker_segments.clear()
        with self.buffer_lock:
            self.audio_buffer.clear()
        self.log_message("🗑️ All transcriptions cleared")
        
    def toggle_recording(self):
        """Toggle recording on/off"""
        if not self.is_recording:
            self.start_recording()
        else:
            self.stop_recording()
            
    def start_recording(self):
        """Start microphone recording and transcription"""
        if self.speech_client is None:
            self.log_message("❌ Speech client not initialized!")
            return
            
        try:
            # Initialize PyAudio
            self.pyaudio_instance = pyaudio.PyAudio()
            
            # Check if microphone is available
            device_count = self.pyaudio_instance.get_device_count()
            self.log_message(f"🎤 Found {device_count} audio devices")
            
            # Open microphone stream
            self.stream = self.pyaudio_instance.open(
                format=self.FORMAT,
                channels=self.CHANNELS,
                rate=self.RATE,
                input=True,
                frames_per_buffer=self.CHUNK,
                stream_callback=self.audio_callback
            )
            
            self.is_recording = True
            self.stream.start_stream()
            
            # Clear audio buffer with thread safety
            with self.buffer_lock:
                self.audio_buffer.clear()
            self.last_speaker_analysis = time.time()
            
            # Update GUI
            self.toggle_button.config(
                text="🛑 Stop Recording",
                bg="#f44336"
            )
            self.status_label.config(
                text="🟢 Recording... Speak now!",
                fg="green"
            )
            self.speaker_status_label.config(
                text="👥 Speaker analysis: Active",
                fg="blue"
            )
            
            # Start transcription thread
            self.transcription_thread = threading.Thread(target=self.transcription_worker)
            self.transcription_thread.daemon = True
            self.transcription_thread.start()
            
            # Start speaker analysis thread
            self.speaker_analysis_thread = threading.Thread(target=self.speaker_analysis_worker)
            self.speaker_analysis_thread.daemon = True
            self.speaker_analysis_thread.start()
            
            self.buffer_duration = int(self.speaker_interval_var.get())
            self.log_message(f"🎙️ Started recording with language: {self.language_var.get()}")
            self.log_message(f"👥 Speaker analysis interval: {self.buffer_duration} seconds")
            
        except Exception as e:
            self.log_message(f"❌ Error starting recording: {e}")
            self.is_recording = False
            
    def stop_recording(self):
        """Stop microphone recording"""
        self.is_recording = False
        
        # Wait for threads to finish
        if self.transcription_thread and self.transcription_thread.is_alive():
            self.log_message("🔄 Waiting for transcription thread to stop...")
            self.transcription_thread.join(timeout=2)
            
        if self.speaker_analysis_thread and self.speaker_analysis_thread.is_alive():
            self.log_message("🔄 Waiting for speaker analysis thread to stop...")
            self.speaker_analysis_thread.join(timeout=2)
        
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            
        if self.pyaudio_instance:
            self.pyaudio_instance.terminate()
            
        # Clear the audio queue
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break
            
        # Update GUI
        self.toggle_button.config(
            text="🎙️ Start Recording",
            bg="#4CAF50"
        )
        self.status_label.config(
            text="🔴 Recording stopped",
            fg="#666"
        )
        self.speaker_status_label.config(
            text="👥 Speaker analysis: Stopped",
            fg="#666"
        )
        
        self.log_message("🛑 Recording stopped")
        
        # Perform final speaker analysis on remaining buffer
        with self.buffer_lock:
            has_buffer = bool(self.audio_buffer)
        
        if has_buffer:
            self.log_message("🔄 Processing final audio buffer for speakers...")
            self.process_speaker_diarization()
        
    def audio_callback(self, in_data, frame_count, time_info, status):
        """Callback function for audio stream"""
        if self.is_recording:
            self.audio_queue.put(in_data)
            # Also store in buffer for speaker analysis with thread safety
            with self.buffer_lock:
                self.audio_buffer.append((time.time(), in_data))
        return (None, pyaudio.paContinue)
                
    def transcription_worker(self):
        """Worker thread for handling Google Cloud Speech streaming (real-time)"""
        retry_count = 0
        max_retries = 3
        
        while self.is_recording and retry_count < max_retries:
            try:
                # Get current language setting (refresh each time to pick up changes)
                current_language = self.language_var.get()
                alternative_language = ["en-US"] if current_language == "th-TH" else ["th-TH"]
                
                self.log_message(f"🌐 Using language: {current_language} (alternative: {alternative_language})")
                
                # Configuration for streaming recognition - use the same language as the GUI setting
                config = speech.RecognitionConfig(
                    encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                    sample_rate_hertz=self.RATE,
                    language_code=current_language,  # Use current language setting
                    alternative_language_codes=alternative_language,
                    enable_automatic_punctuation=True,
                    model="default",  # Use default model for better Thai support
                )
                
                streaming_config = speech.StreamingRecognitionConfig(
                    config=config,
                    interim_results=True,
                    single_utterance=False,
                )
                
                self.log_message("🚀 Starting real-time streaming recognition...")
                
                # Create the generator function for requests
                def request_generator():
                    # Subsequent requests contain audio data
                    request_count = 0
                    while self.is_recording:
                        try:
                            # Get audio data from queue with timeout
                            chunk = self.audio_queue.get(timeout=0.5)  # Shorter timeout
                            request_count += 1

                            if request_count % 50 == 0:  # Log every 50 requests
                                self.log_message(f"🔍 Sent {request_count} audio requests to streaming API")
                            
                            yield speech.StreamingRecognizeRequest(audio_content=chunk)
                        except queue.Empty:
                            # Continue but don't send empty requests for streaming
                            continue
                
                # Perform streaming recognition
                responses = self.speech_client.streaming_recognize(
                    config=streaming_config,
                    requests=request_generator()
                )
                
                self.log_message("✅ Streaming recognition connected, listening for responses...")
                response_count = 0
                
                # Process responses
                for response in responses:
                    if not self.is_recording:
                        break
                    
                    response_count += 1
                        
                    # Debug: Log when we receive a response
                    if response.results:
                        self.log_message(f"🔍 Received streaming response with {len(response.results)} results")
                        
                        for result in response.results:
                            transcript = result.alternatives[0].transcript
                            confidence = result.alternatives[0].confidence if result.alternatives[0].confidence else 0.0
                            
                            # Debug: Log transcript details
                            self.log_message(f"🔍 Transcript: '{transcript}' (final: {result.is_final})")
                            
                            if result.is_final:
                                self.log_message(f"✅ Final: {transcript} (confidence: {confidence:.2f})")
                                self.add_transcription(transcript, is_final=True)
                            else:
                                self.add_transcription(transcript, is_final=False)
                    else:
                        # Empty response - log occasionally
                        if response_count % 20 == 0:
                            self.log_message(f"🔍 Empty response #{response_count} received (waiting for speech)")
                
                # If we get here without error, reset retry count
                retry_count = 0
                                
            except Exception as e:
                error_msg = str(e)
                self.log_message(f"❌ Real-time transcription error: {error_msg}")
                
                if "400" in error_msg and "Audio Timeout" in error_msg:
                    self.log_message("🔄 Audio timeout - restarting streaming recognition...")
                    retry_count += 1
                    if retry_count < max_retries:
                        # Clear the audio queue and restart
                        while not self.audio_queue.empty():
                            try:
                                self.audio_queue.get_nowait()
                            except queue.Empty:
                                break
                        time.sleep(1)
                        continue
                
                if self.is_recording:
                    self.log_message(f"🛑 Stopping recording due to transcription error (retry {retry_count}/{max_retries})")
                    if retry_count >= max_retries:
                        self.stop_recording()
                break
                
    def speaker_analysis_worker(self):
        """Worker thread for periodic speaker diarization analysis"""
        while self.is_recording:
            try:
                current_time = time.time()
                
                # Check if it's time for speaker analysis
                if current_time - self.last_speaker_analysis >= self.buffer_duration:
                    with self.buffer_lock:
                        buffer_size = len(self.audio_buffer)
                        has_buffer = bool(self.audio_buffer)
                    
                    if has_buffer:
                        self.log_message(f"🔄 Analyzing speakers (buffer: {buffer_size} chunks)...")
                        self.process_speaker_diarization()
                        self.last_speaker_analysis = current_time
                
                time.sleep(1)  # Check every second
                
            except Exception as e:
                self.log_message(f"❌ Speaker analysis error: {e}")
                
    def process_speaker_diarization(self):
        """Process only the recent audio segment for speaker diarization using Beta API"""
        try:
            # Calculate how many chunks represent the analysis duration
            chunks_per_second = self.RATE / self.CHUNK
            max_chunks = int(self.buffer_duration * chunks_per_second)
            
            # Create a safe copy of only the recent audio segment
            with self.buffer_lock:
                if not self.audio_buffer:
                    return
                
                # Get only the most recent chunks (last N seconds)
                if len(self.audio_buffer) > max_chunks:
                    # Take only the last max_chunks
                    recent_buffer = list(self.audio_buffer)[-max_chunks:]
                else:
                    # Take all available chunks if less than max
                    recent_buffer = list(self.audio_buffer)
                
                self.log_message(f"🔍 Processing last {len(recent_buffer)} chunks (target: {max_chunks} for {self.buffer_duration}s)")
                
            # Create temporary WAV file from recent buffer only
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_filename = temp_file.name
                
            # Write only recent audio buffer to WAV file
            with wave.open(temp_filename, 'wb') as wf:
                wf.setnchannels(self.CHANNELS)
                wf.setsampwidth(self.pyaudio_instance.get_sample_size(self.FORMAT))
                wf.setframerate(self.RATE)
                
                # Write only the recent buffered audio
                for _, audio_data in recent_buffer:
                    wf.writeframes(audio_data)
            
            # Check the created file
            with wave.open(temp_filename, 'rb') as wf:
                duration = wf.getnframes() / wf.getframerate()
                self.log_message(f"📊 Created audio file: {duration:.1f}s, {wf.getframerate()}Hz, {wf.getnchannels()}ch")
            
            # Check if audio is too long for sync API (1 minute limit)
            if duration > 58:  # Leave 2 seconds buffer
                self.log_message(f"⚠️ Audio too long ({duration:.1f}s) for sync API, skipping this segment")
                os.unlink(temp_filename)
                
                # Clear most of the buffer but keep some recent audio
                with self.buffer_lock:
                    recent_chunks = int(30 * self.RATE / self.CHUNK)  # Keep last 30 seconds
                    if len(self.audio_buffer) > recent_chunks:
                        # Remove old audio, keep recent
                        for _ in range(len(self.audio_buffer) - recent_chunks):
                            self.audio_buffer.popleft()
                
                self.speaker_status_label.config(text="👥 Speaker analysis: Active (Beta API)", fg="blue")
                return
            
            # Get beta API credentials
            access_token = self.get_beta_api_credentials()
            if not access_token:
                self.log_message("❌ Failed to get beta API credentials")
                return
            
            # Read and encode audio file
            with open(temp_filename, "rb") as audio_file:
                audio_content = base64.b64encode(audio_file.read()).decode('utf-8')
            
            # Get current language setting
            current_language = self.language_var.get()
            self.log_message(f"🌐 Beta API using language: {current_language}")
            
            # Prepare beta API request using the working configuration
            url = "https://speech.googleapis.com/v1p1beta1/speech:recognize"
            headers = {
                'Authorization': f'Bearer {access_token}',
                'Content-Type': 'application/json',
            }
            
            request_body = {
                "audio": {
                    "content": audio_content
                },
                "config": {
                    "enableAutomaticPunctuation": True,
                    "enableSpeakerDiarization": True,
                    "encoding": "LINEAR16",
                    "sampleRateHertz": self.RATE,  # Add sample rate for microphone audio
                    "languageCode": current_language,  # Use current language setting
                    "model": "default",
                    "diarizationConfig": {
                        "enableSpeakerDiarization": True,
                        "minSpeakerCount": 1,
                        "maxSpeakerCount": 4  # Allow up to 4 speakers, let API auto-detect
                    }
                }
            }
            
            # Update status
            self.speaker_status_label.config(text="👥 Analyzing speakers (Beta API)...", fg="orange")
            self.root.update_idletasks()
            
            # Make beta API call
            response = requests.post(url, headers=headers, data=json.dumps(request_body))
            
            if response.status_code != 200:
                error_details = ""
                try:
                    error_data = response.json()
                    error_details = f": {error_data.get('error', {}).get('message', 'Unknown error')}"
                except:
                    error_details = f": {response.text[:200]}"
                
                self.log_message(f"❌ Beta API request failed: {response.status_code}{error_details}")
                self.speaker_status_label.config(text="👥 Speaker analysis: Error", fg="red")
                return
            
            result = response.json()
            
            # Process beta API results
            if 'results' not in result or not result['results']:
                self.log_message("⚠️ No speech detected in this segment")
                self.speaker_status_label.config(text="👥 Speaker analysis: No speech", fg="gray")
                return
            
            # Extract word-level speaker information
            words_info = []
            for result_item in result['results']:
                if 'alternatives' in result_item and result_item['alternatives']:
                    alternative = result_item['alternatives'][0]
                    if 'words' in alternative:
                        words_info.extend(alternative['words'])
            
            if not words_info:
                self.log_message("⚠️ No speaker diarization data found")
                return
            
            # Debug: Log word-level speaker information to understand the issue
            self.log_message(f"🔍 Debug: Found {len(words_info)} words with speaker tags")
            for i, word_info in enumerate(words_info[:5]):  # Show first 5 words for debugging
                word = word_info.get('word', '')
                speaker_tag = word_info.get('speakerTag', 1)
                start_time = word_info.get('startTime', '0s')
                self.log_message(f"🔍 Word {i+1}: '{word}' -> Speaker {speaker_tag} at {start_time}")
            
            # Filter out duplicate words with same timing but different speakers
            # This happens when the API incorrectly assigns the same word to multiple speakers
            filtered_words = []
            seen_words = {}  # Track word+time combinations
            
            for word_info in words_info:
                word = word_info.get('word', '')
                start_time = word_info.get('startTime', '0s')
                speaker_tag = word_info.get('speakerTag', 1)
                
                # Create a unique key for this word at this time
                word_time_key = f"{word}_{start_time}"
                
                if word_time_key not in seen_words:
                    # First occurrence of this word at this time
                    seen_words[word_time_key] = word_info
                    filtered_words.append(word_info)
                else:
                    # Duplicate detected - keep the one with lower speaker tag (more reliable)
                    existing_speaker = seen_words[word_time_key].get('speakerTag', 1)
                    if speaker_tag < existing_speaker:
                        # Replace with lower speaker tag
                        seen_words[word_time_key] = word_info
                        # Remove the old one and add the new one
                        filtered_words = [w for w in filtered_words if not (
                            w.get('word') == word and w.get('startTime') == start_time
                        )]
                        filtered_words.append(word_info)
            
            self.log_message(f"🔍 After filtering: {len(filtered_words)} unique words (removed {len(words_info) - len(filtered_words)} duplicates)")
            words_info = filtered_words
            
            # Group words by speaker for conversation view
            conversations = []
            current_speaker = None
            current_sentence = ""
            current_start_time = None
            current_end_time = None
            
            for word_info in words_info:
                word = word_info.get('word', '')
                speaker_tag = word_info.get('speakerTag', 1)
                start_time = float(word_info.get('startTime', '0s').rstrip('s'))
                end_time = float(word_info.get('endTime', '0s').rstrip('s'))
                
                if current_speaker is None:
                    current_speaker = speaker_tag
                    current_start_time = start_time
                
                # If speaker changes or we detect end of sentence
                if (speaker_tag != current_speaker or 
                    word.endswith('.') or word.endswith('!') or word.endswith('?') or
                    (start_time - current_end_time > 1.0 if current_end_time else False)):
                    
                    if current_sentence.strip():
                        conversations.append({
                            'speaker': current_speaker,
                            'text': current_sentence.strip(),
                            'start_time': current_start_time,
                            'end_time': current_end_time
                        })
                    
                    current_sentence = ""
                    current_speaker = speaker_tag
                    current_start_time = start_time
                
                current_sentence += word + " "
                current_end_time = end_time
            
            # Add the last sentence
            if current_sentence.strip():
                conversations.append({
                    'speaker': current_speaker,
                    'text': current_sentence.strip(),
                    'start_time': current_start_time,
                    'end_time': current_end_time
                })
            
            # Format speaker analysis output
            duration = len(recent_buffer) * self.CHUNK / self.RATE
            unique_speakers = set(conv['speaker'] for conv in conversations)
            
            analysis_output = f"🎤 Speaker Analysis Results (Beta API):\n"
            analysis_output += f"📊 Duration: {duration:.1f} seconds (recent segment)\n"
            analysis_output += f"👥 Speakers detected: {len(unique_speakers)}\n"
            analysis_output += f"📝 Conversation segments: {len(conversations)}\n\n"
            
            # Show conversation segments
            for conv in conversations:
                analysis_output += f"[{conv['start_time']:.1f}-{conv['end_time']:.1f}s] "
                analysis_output += f"Speaker {conv['speaker']}: {conv['text']}\n"
            
            # Show speaker summary
            analysis_output += f"\n� Speaker Summary:\n"
            for speaker in sorted(unique_speakers):
                speaker_segments = [conv for conv in conversations if conv['speaker'] == speaker]
                total_time = sum(conv['end_time'] - conv['start_time'] for conv in speaker_segments)
                analysis_output += f"Speaker {speaker}: {len(speaker_segments)} segments, {total_time:.1f}s speaking time\n"
            
            self.add_speaker_analysis(analysis_output)
            self.log_message(f"✅ Beta API speaker analysis complete: {len(unique_speakers)} speakers, {len(conversations)} segments")
            
            # Clean up temporary file
            os.unlink(temp_filename)
            
            # Trim the audio buffer to keep only recent audio (prevent memory buildup)
            with self.buffer_lock:
                # Keep only the last 30 seconds of audio to prevent memory issues
                max_buffer_chunks = int(30 * self.RATE / self.CHUNK)  # 30 seconds
                if len(self.audio_buffer) > max_buffer_chunks:
                    # Remove old chunks, keep recent ones
                    excess_chunks = len(self.audio_buffer) - max_buffer_chunks
                    for _ in range(excess_chunks):
                        self.audio_buffer.popleft()
                    self.log_message(f"🧹 Trimmed buffer: removed {excess_chunks} old chunks, kept {len(self.audio_buffer)} recent ones")
            
            self.speaker_status_label.config(text="👥 Speaker analysis: Active (Beta API)", fg="blue")
            
        except Exception as e:
            self.log_message(f"❌ Beta API speaker diarization error: {e}")
            self.speaker_status_label.config(text="👥 Speaker analysis: Error", fg="red")
            # Clean up temporary file on error
            try:
                if 'temp_filename' in locals():
                    os.unlink(temp_filename)
            except:
                pass
                
    def run(self):
        """Start the GUI application"""
        self.log_message("🎉 Microphone Streaming with Speaker ID started!")
        self.log_message("� Using Google Cloud Speech v1p1beta1 REST API for enhanced speaker diarization")
        self.log_message("�💡 Click 'Start Recording' to begin real-time transcription")
        self.log_message("👥 Speaker analysis will run every few seconds using Beta API")
        
        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            self.log_message("👋 Application terminated by user")
        finally:
            if self.is_recording:
                self.stop_recording()

def check_dependencies():
    """Check if required dependencies are installed"""
    try:
        import pyaudio
        return True
    except ImportError:
        print("❌ PyAudio is not installed!")
        print("📦 Please install it with: pip install pyaudio")
        return False

if __name__ == "__main__":
    print("🎤 Real-time Microphone Streaming with Speaker Identification")
    print("=" * 60)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
        
    # Check service account file
    if not os.path.exists("voice-sun-1-67f7efc777f3.json"):
        print("❌ Service account file 'voice-sun-1-67f7efc777f3.json' not found!")
        sys.exit(1)
        
    # Start the application
    app = MicrophoneStreamingWithSpeakers()
    app.run()

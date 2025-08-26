#!/usr/bin/env python3
"""
Authenticated Hybrid Speech Recognition + Speaker Diarization
Combines Google Speech-to-Text with pyannote-audio using proper HF authentication
"""

import os
import sys
import time
import threading
import queue
import tempfile
import wave
import json
import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog, messagebox
import pyaudio
import numpy as np
from google.cloud import speech
from google.oauth2 import service_account

# Hugging Face authentication
try:
    from huggingface_hub import login, HfFolder
    import torch
    from pyannote.audio import Pipeline
    PYANNOTE_AVAILABLE = True
    
    # Check if user is logged in
    try:
        token = HfFolder.get_token()
        if token:
            print("✅ Hugging Face authentication found")
        else:
            print("⚠️ Hugging Face authentication not found - run: huggingface-cli login")
    except:
        print("⚠️ Hugging Face authentication not found - run: huggingface-cli login")
        
except ImportError as e:
    PYANNOTE_AVAILABLE = False
    print(f"❌ pyannote-audio not available: {e}")

class AuthenticatedHybridDiarization:
    def __init__(self):
        # Audio configuration
        self.RATE = 16000
        self.CHUNK = 1024
        self.CHANNELS = 1
        self.FORMAT = pyaudio.paInt16
        
        # Threading and control
        self.is_recording = False
        self.audio_queue = queue.Queue()
        self.pyaudio_instance = None
        self.stream = None
        self.processing_thread = None
        
        # Services
        self.speech_client = None
        self.diarization_pipeline = None
        
        # Data tracking
        self.segment_counter = 0
        self.speaker_stats = {}
        
        # Persistent speaker tracking across segments
        self.global_speaker_mapping = {}  # Maps pyannote labels to consistent global IDs
        self.next_global_speaker_id = 0
        self.speaker_voice_profiles = {}  # Store voice characteristics for each global speaker
        
        # Setup GUI and services
        self.setup_gui()
        self.setup_services()
        
    def setup_gui(self):
        """Create the main interface"""
        self.root = tk.Tk()
        self.root.title("🔐 Authenticated Hybrid Speech + Speaker Diarization")
        self.root.geometry("1100x850")
        self.root.configure(bg="#f0f0f0")
        
        # Title
        title_label = tk.Label(
            self.root, 
            text="🔐 Authenticated Hybrid: Google Speech + pyannote Diarization", 
            font=("Arial", 16, "bold"),
            bg="#f0f0f0", fg="#333"
        )
        title_label.pack(pady=10)
        
        # Authentication status frame
        auth_frame = tk.Frame(self.root, bg="white", relief="solid", bd=1)
        auth_frame.pack(fill=tk.X, padx=20, pady=(0, 10))
        
        auth_title = tk.Label(auth_frame, text="🔐 Authentication Status", 
                             font=("Arial", 12, "bold"), bg="white")
        auth_title.pack(pady=5)
        
        auth_status_frame = tk.Frame(auth_frame, bg="white")
        auth_status_frame.pack(pady=5)
        
        self.google_status = tk.Label(auth_status_frame, text="Google: Checking...", 
                                     bg="white", font=("Arial", 10))
        self.google_status.pack(side=tk.LEFT, padx=20)
        
        self.hf_status = tk.Label(auth_status_frame, text="HuggingFace: Checking...", 
                                 bg="white", font=("Arial", 10))
        self.hf_status.pack(side=tk.LEFT, padx=20)
        
        self.pyannote_status = tk.Label(auth_status_frame, text="pyannote: Checking...", 
                                       bg="white", font=("Arial", 10))
        self.pyannote_status.pack(side=tk.LEFT, padx=20)
        
        # Controls
        control_frame = tk.Frame(self.root, bg="#f0f0f0")
        control_frame.pack(pady=10)
        
        self.toggle_button = tk.Button(
            control_frame, text="🎙️ Start Recording",
            font=("Arial", 12, "bold"), bg="#4CAF50", fg="white",
            padx=20, pady=10, command=self.toggle_recording
        )
        self.toggle_button.pack(side=tk.LEFT, padx=10)
        
        clear_button = tk.Button(
            control_frame, text="🗑️ Clear",
            font=("Arial", 12), bg="#f44336", fg="white",
            padx=20, pady=10, command=self.clear_all
        )
        clear_button.pack(side=tk.LEFT, padx=10)
        
        reset_speakers_button = tk.Button(
            control_frame, text="🔄 Reset Speakers",
            font=("Arial", 12), bg="#ff9800", fg="white",
            padx=20, pady=10, command=self.reset_speakers
        )
        reset_speakers_button.pack(side=tk.LEFT, padx=10)
        
        upload_button = tk.Button(
            control_frame, text="📁 Upload Audio File",
            font=("Arial", 12), bg="#9c27b0", fg="white",
            padx=20, pady=10, command=self.upload_audio_file
        )
        upload_button.pack(side=tk.LEFT, padx=10)
        
        # Language selection
        lang_frame = tk.Frame(control_frame, bg="#f0f0f0")
        lang_frame.pack(side=tk.LEFT, padx=20)
        
        tk.Label(lang_frame, text="Language:", bg="#f0f0f0").pack()
        self.language_var = tk.StringVar(value="th-TH")
        language_combo = ttk.Combobox(
            lang_frame, textvariable=self.language_var,
            values=["th-TH", "en-US"], width=8, state="readonly"
        )
        language_combo.pack()
        
        # Speaker statistics
        stats_frame = tk.Frame(self.root, bg="white", relief="solid", bd=1)
        stats_frame.pack(fill=tk.X, padx=20, pady=(10, 5))
        
        self.stats_label = tk.Label(
            stats_frame, text="👥 Speaker Statistics: Waiting for speech...",
            font=("Arial", 11, "bold"), bg="white", fg="#333", anchor="w"
        )
        self.stats_label.pack(fill=tk.X, padx=10, pady=8)
        
        # Main notebook
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill=tk.BOTH, expand=True, padx=20, pady=5)
        
        # Results tab
        results_frame = ttk.Frame(notebook)
        notebook.add(results_frame, text="🎙️ Live Results")
        
        self.transcript_text = scrolledtext.ScrolledText(
            results_frame, height=15, font=("Arial", 11),
            wrap=tk.WORD, bg="white", fg="#333"
        )
        self.transcript_text.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Details tab
        details_frame = ttk.Frame(notebook)
        notebook.add(details_frame, text="🔍 Processing Details")
        
        # Create paned window for side-by-side view
        paned = ttk.PanedWindow(details_frame, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Google results
        google_frame = ttk.LabelFrame(paned, text="Google Speech Recognition")
        paned.add(google_frame, weight=1)
        
        self.google_text = scrolledtext.ScrolledText(
            google_frame, height=12, font=("Arial", 10), bg="#f8f8f8"
        )
        self.google_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # pyannote results
        pyannote_frame = ttk.LabelFrame(paned, text="pyannote Speaker Diarization")
        paned.add(pyannote_frame, weight=1)
        
        self.pyannote_text = scrolledtext.ScrolledText(
            pyannote_frame, height=12, font=("Arial", 10), bg="#f8f8f8"
        )
        self.pyannote_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Log
        log_label = tk.Label(self.root, text="📋 System Log:",
                            font=("Arial", 10, "bold"), bg="#f0f0f0", anchor="w")
        log_label.pack(fill=tk.X, padx=20, pady=(5, 2))
        
        self.log_text = scrolledtext.ScrolledText(
            self.root, height=6, font=("Arial", 9),
            wrap=tk.WORD, bg="#f8f8f8", fg="#666"
        )
        self.log_text.pack(fill=tk.X, padx=20, pady=(0, 20))
        
    def setup_services(self):
        """Initialize Google and pyannote services with authentication"""
        # Setup Google Cloud
        try:
            service_account_path = "voice-sun-1-67f7efc777f3.json"
            if os.path.exists(service_account_path):
                credentials = service_account.Credentials.from_service_account_file(
                    service_account_path,
                    scopes=["https://www.googleapis.com/auth/cloud-platform"]
                )
                self.speech_client = speech.SpeechClient(credentials=credentials)
                self.google_status.config(text="Google: ✅ Ready", fg="green")
                self.log_message("✅ Google Cloud Speech initialized")
            else:
                self.google_status.config(text="Google: ❌ No credentials", fg="red")
                self.log_message("❌ Google credentials not found")
        except Exception as e:
            self.google_status.config(text="Google: ❌ Error", fg="red")
            self.log_message(f"❌ Google setup error: {e}")
            
        # Check Hugging Face authentication
        try:
            from huggingface_hub import HfFolder
            token = HfFolder.get_token()
            if token:
                self.hf_status.config(text="HuggingFace: ✅ Authenticated", fg="green")
                self.log_message("✅ Hugging Face authentication verified")
            else:
                self.hf_status.config(text="HuggingFace: ❌ Not logged in", fg="red")
                self.log_message("❌ Hugging Face not authenticated - run: huggingface-cli login")
        except Exception as e:
            self.hf_status.config(text="HuggingFace: ❌ Error", fg="red")
            self.log_message(f"❌ HuggingFace check error: {e}")
            
        # Setup pyannote with authentication
        if PYANNOTE_AVAILABLE:
            try:
                self.pyannote_status.config(text="pyannote: 🔄 Loading...", fg="orange")
                self.log_message("🔄 Loading pyannote diarization pipeline...")
                self.root.update()
                
                # Try to load with authentication
                try:
                    # First try with stored token
                    self.diarization_pipeline = Pipeline.from_pretrained(
                        "pyannote/speaker-diarization-3.1",
                        use_auth_token=True  # Use stored HF token
                    )
                except Exception as auth_error:
                    self.log_message(f"⚠️ Auth token method failed: {auth_error}")
                    # Fallback to default method
                    self.diarization_pipeline = Pipeline.from_pretrained(
                        "pyannote/speaker-diarization-3.1"
                    )
                
                # Use GPU if available
                if torch.cuda.is_available():
                    self.diarization_pipeline = self.diarization_pipeline.to(torch.device("cuda"))
                    self.pyannote_status.config(text="pyannote: ✅ Ready (GPU)", fg="green")
                    self.log_message("✅ pyannote loaded with GPU acceleration")
                else:
                    self.pyannote_status.config(text="pyannote: ✅ Ready (CPU)", fg="green")
                    self.log_message("✅ pyannote loaded with CPU")
                    
            except Exception as e:
                self.pyannote_status.config(text="pyannote: ❌ Error", fg="red")
                self.log_message(f"❌ pyannote setup error: {e}")
                self.log_message("💡 Solutions:")
                self.log_message("   1. Run: huggingface-cli login")
                self.log_message("   2. Accept license: https://huggingface.co/pyannote/speaker-diarization-3.1")
        else:
            self.pyannote_status.config(text="pyannote: ❌ Not installed", fg="red")
            self.log_message("❌ pyannote-audio not available")
            
    def log_message(self, message):
        """Add message to log"""
        timestamp = time.strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}\n"
        self.log_text.insert(tk.END, log_entry)
        self.log_text.see(tk.END)
        self.root.update_idletasks()
        print(log_entry.strip())
        
    def clear_all(self):
        """Clear all displays"""
        self.transcript_text.delete(1.0, tk.END)
        self.google_text.delete(1.0, tk.END)
        self.pyannote_text.delete(1.0, tk.END)
        self.speaker_stats.clear()
        self.update_speaker_stats()
        self.log_message("🗑️ All displays cleared")
        
    def update_speaker_stats(self):
        """Update speaker statistics display"""
        if not self.speaker_stats:
            self.stats_label.config(text="👥 Speaker Statistics: No speakers detected")
            return
            
        total_words = sum(self.speaker_stats.values())
        stats_text = f"👥 Speaker Statistics: "
        
        for speaker_id in sorted(self.speaker_stats.keys()):
            word_count = self.speaker_stats[speaker_id]
            if total_words > 0:
                percentage = (word_count / total_words) * 100
                stats_text += f"Speaker {speaker_id}: {word_count} words ({percentage:.1f}%) | "
            else:
                stats_text += f"Speaker {speaker_id}: {word_count} words | "
                
        # Remove trailing " | "
        stats_text = stats_text.rstrip(" | ")
        self.stats_label.config(text=stats_text)
        
    def reset_speakers(self):
        """Reset speaker tracking - all speakers will be re-detected"""
        self.global_speaker_mapping.clear()
        self.next_global_speaker_id = 0
        self.speaker_voice_profiles.clear()
        self.speaker_stats.clear()
        self.log_message("🔄 Speaker tracking reset - next speakers will get new IDs starting from 0")
        
    def upload_audio_file(self):
        """Upload and process an audio file with pyannote"""
        if not self.diarization_pipeline:
            messagebox.showerror("Error", "pyannote pipeline not ready. Please wait for initialization.")
            return
            
        # File selection dialog
        file_path = filedialog.askopenfilename(
            title="Select Audio File",
            filetypes=[
                ("Audio Files", "*.wav *.mp3 *.flac *.m4a *.ogg"),
                ("WAV files", "*.wav"),
                ("MP3 files", "*.mp3"),
                ("FLAC files", "*.flac"),
                ("All files", "*.*")
            ]
        )
        
        if not file_path:
            return
            
        # Clear previous results
        self.clear_all()
        self.reset_speakers()
        
        self.log_message(f"📁 Processing uploaded file: {os.path.basename(file_path)}")
        
        # Process in separate thread to avoid freezing GUI
        threading.Thread(target=self.process_uploaded_file, args=(file_path,), daemon=True).start()
        
    def process_uploaded_file(self, file_path):
        """Process uploaded audio file"""
        try:
            # Update status
            self.log_message("🔄 Analyzing audio file...")
            self.root.update()
            
            # Process with Google Speech (if supported format)
            google_results = []
            if file_path.lower().endswith('.wav'):
                try:
                    google_results = self.process_google(file_path)
                    self.log_message(f"🗣️ Google: {len(google_results)} transcription results")
                except Exception as e:
                    self.log_message(f"⚠️ Google processing failed: {e}")
                    # Show more helpful error message
                    if "sample_rate_hertz" in str(e):
                        self.log_message("💡 Try converting the audio to 16kHz WAV format for Google Speech compatibility")
            else:
                self.log_message("⚠️ Google Speech only supports WAV files - skipping transcription")
            
            # Process with pyannote (supports multiple formats)
            self.log_message("👤 Running pyannote speaker diarization...")
            pyannote_results = self.process_pyannote(file_path)
            
            # Generate summary
            if pyannote_results:
                unique_speakers = set(speaker_id for _, _, speaker_id in pyannote_results)
                total_duration = max(end_time for _, end_time, _ in pyannote_results) if pyannote_results else 0
                
                self.log_message("=" * 50)
                self.log_message("📊 AUDIO FILE ANALYSIS SUMMARY")
                self.log_message("=" * 50)
                self.log_message(f"📁 File: {os.path.basename(file_path)}")
                self.log_message(f"⏱️ Duration: {total_duration:.1f} seconds")
                self.log_message(f"👥 Speakers detected: {len(unique_speakers)}")
                self.log_message(f"🎯 Speaker segments: {len(pyannote_results)}")
                
                # Speaker time breakdown
                speaker_times = {}
                for start_time, end_time, speaker_id in pyannote_results:
                    duration = end_time - start_time
                    speaker_times[speaker_id] = speaker_times.get(speaker_id, 0) + duration
                
                for speaker_id in sorted(speaker_times.keys()):
                    duration = speaker_times[speaker_id]
                    percentage = (duration / total_duration * 100) if total_duration > 0 else 0
                    self.log_message(f"🎤 Speaker {speaker_id}: {duration:.1f}s ({percentage:.1f}%)")
                
                self.log_message("=" * 50)
                
                # Combine with Google results if available
                if google_results:
                    combined_results = self.combine_results(google_results, pyannote_results)
                    self.display_results(combined_results)
                else:
                    # Display pyannote-only results
                    self.display_pyannote_only_results(pyannote_results, file_path)
                    
            else:
                self.log_message("❌ No speakers detected in the audio file")
                
        except Exception as e:
            self.log_message(f"❌ Error processing file: {e}")
            messagebox.showerror("Processing Error", f"Failed to process audio file:\n{e}")
            
    def display_pyannote_only_results(self, pyannote_results, file_path):
        """Display results when only pyannote data is available"""
        timestamp = time.strftime("%H:%M:%S")
        
        # Group by speaker
        speaker_segments = {}
        for start_time, end_time, speaker_id in pyannote_results:
            if speaker_id not in speaker_segments:
                speaker_segments[speaker_id] = []
            speaker_segments[speaker_id].append((start_time, end_time))
        
        # Display in transcript area with speaker colors
        self.transcript_text.insert(tk.END, f"[{timestamp}] 📁 Audio File: {os.path.basename(file_path)}\n")
        self.transcript_text.insert(tk.END, f"👥 {len(speaker_segments)} speakers detected by pyannote\n\n")
        
        # Speaker colors
        speaker_colors = {0: "#e74c3c", 1: "#3498db", 2: "#2ecc71", 3: "#f39c12", 4: "#9b59b6", 5: "#1abc9c"}
        
        # Create timeline of speaker changes
        all_segments = []
        for speaker_id, segments in speaker_segments.items():
            for start, end in segments:
                all_segments.append((start, end, speaker_id))
        
        # Sort by start time
        all_segments.sort(key=lambda x: x[0])
        
        # Display timeline
        for i, (start, end, speaker_id) in enumerate(all_segments):
            duration = end - start
            color = speaker_colors.get(speaker_id, "#333333")
            
            # Insert speaker timeline entry
            start_pos = self.transcript_text.index(tk.END)
            text = f"[{start:.1f}s-{end:.1f}s] Speaker {speaker_id}: {duration:.1f}s of speech\n"
            self.transcript_text.insert(tk.END, text)
            end_pos = self.transcript_text.index(tk.END + "-1c")
            
            # Apply color formatting
            self.transcript_text.tag_add(f"speaker_{speaker_id}_file", start_pos, end_pos)
            self.transcript_text.tag_config(f"speaker_{speaker_id}_file", 
                                           foreground=color, font=("Arial", 11, "bold"))
        
        self.transcript_text.insert(tk.END, "\n💡 Note: Transcription not available - only speaker diarization shown\n")
        self.transcript_text.insert(tk.END, "💡 For transcription, ensure audio is 16kHz WAV format and language is set correctly\n\n")
        self.transcript_text.see(tk.END)
        
        # Update speaker stats (use segment count as approximation)
        for speaker_id in speaker_segments:
            segments = speaker_segments[speaker_id]
            total_duration = sum(end - start for start, end in segments)
            # Estimate words (rough approximation: 2 words per second)
            estimated_words = max(1, int(total_duration * 2))
            self.speaker_stats[speaker_id] = estimated_words
            
        # Update the statistics display
        self.update_speaker_stats()
        
    def display_google_only_results(self, google_results):
        """Display results when only Google has transcription (no speakers detected)"""
        timestamp = time.strftime("%H:%M:%S")
        
        for result in google_results:
            transcript = result['transcript']
            confidence = result.get('confidence', 0.0)
            
            # Assign to default Speaker 0
            self.speaker_stats[0] = self.speaker_stats.get(0, 0) + len(transcript.split())
            
            # Display with default speaker
            text = f"[{timestamp}] [Speaker 0] {transcript} ({confidence:.0%})\n"
            self.transcript_text.insert(tk.END, text)
            
            # Apply default speaker color
            start_index = self.transcript_text.index(f"end-{len(text)}c")
            end_index = self.transcript_text.index("end-1c")
            self.transcript_text.tag_add("speaker_0_default", start_index, end_index)
            self.transcript_text.tag_config("speaker_0_default", 
                                           foreground="#e74c3c", font=("Arial", 11, "bold"))
            
        self.transcript_text.see(tk.END)
        self.update_speaker_stats()
        
    def display_pyannote_only_live_results(self, pyannote_results):
        """Display live results when only pyannote detects speakers (no transcription)"""
        print(f"🎯 DEBUG: display_pyannote_only_live_results called with {len(pyannote_results)} results")
        timestamp = time.strftime("%H:%M:%S")
        
        # Group by speaker and show active speakers
        active_speakers = set(speaker_id for _, _, speaker_id in pyannote_results)
        print(f"🎯 DEBUG: Active speakers detected: {active_speakers}")
        
        if active_speakers:
            speaker_list = ", ".join(f"Speaker {sid}" for sid in sorted(active_speakers))
            duration = max(end - start for start, end, _ in pyannote_results) if pyannote_results else 0
            
            # Show speaker activity without transcription
            text = f"[{timestamp}] 🎤 Active: {speaker_list} (speaking detected, {duration:.1f}s)\n"
            print(f"🎯 DEBUG: Inserting text: {text.strip()}")
            self.transcript_text.insert(tk.END, text)
            print(f"🎯 DEBUG: Text inserted, current text widget size: {self.transcript_text.index(tk.END)}")
            
            # Update speaker stats (estimate words based on duration)
            for speaker_id in active_speakers:
                speaker_duration = sum(end - start for start, end, sid in pyannote_results if sid == speaker_id)
                estimated_words = max(1, int(speaker_duration * 2))  # ~2 words per second
                self.speaker_stats[speaker_id] = self.speaker_stats.get(speaker_id, 0) + estimated_words
                print(f"🎯 DEBUG: Updated speaker {speaker_id} stats: +{estimated_words} words, total: {self.speaker_stats[speaker_id]}")
                
            # Apply neutral color for speaker activity
            start_index = self.transcript_text.index(f"end-{len(text)}c")
            end_index = self.transcript_text.index("end-1c")
            self.transcript_text.tag_add("speaker_activity", start_index, end_index)
            self.transcript_text.tag_config("speaker_activity", 
                                           foreground="#666666", font=("Arial", 11, "italic"))
            print(f"🎯 DEBUG: Applied styling from {start_index} to {end_index}")
        
        self.transcript_text.see(tk.END)
        self.update_speaker_stats()
        print(f"🎯 DEBUG: GUI updates completed")
        
    def toggle_recording(self):
        """Toggle recording on/off"""
        if not self.is_recording:
            self.start_recording()
        else:
            self.stop_recording()
            
    def start_recording(self):
        """Start recording"""
        if not self.speech_client:
            self.log_message("❌ Google Speech client not ready")
            return
            
        if not self.diarization_pipeline:
            self.log_message("❌ pyannote pipeline not ready")
            return
            
        try:
            self.pyaudio_instance = pyaudio.PyAudio()
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
            
            self.toggle_button.config(text="🛑 Stop Recording", bg="#f44336")
            
            # Start processing thread
            self.processing_thread = threading.Thread(target=self.processing_worker)
            self.processing_thread.daemon = True
            self.processing_thread.start()
            
            self.log_message("🎙️ Recording started - speak now!")
            
        except Exception as e:
            self.log_message(f"❌ Recording start error: {e}")
            
    def stop_recording(self):
        """Stop recording"""
        self.is_recording = False
        
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            
        if self.pyaudio_instance:
            self.pyaudio_instance.terminate()
            
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=2)
            
        self.toggle_button.config(text="🎙️ Start Recording", bg="#4CAF50")
        self.log_message("🛑 Recording stopped")
        
    def audio_callback(self, in_data, frame_count, time_info, status):
        """Callback for audio stream"""
        if self.is_recording:
            self.audio_queue.put(in_data)
        return (None, pyaudio.paContinue)
        
    def processing_worker(self):
        """Process audio chunks with both Google and pyannote"""
        audio_buffer = []
        buffer_duration = 5  # seconds
        max_chunks = int(buffer_duration * self.RATE / self.CHUNK)
        
        while self.is_recording:
            try:
                # Collect audio data
                while len(audio_buffer) < max_chunks and self.is_recording:
                    try:
                        chunk = self.audio_queue.get(timeout=0.1)
                        audio_buffer.append(chunk)
                    except queue.Empty:
                        continue
                        
                if not audio_buffer:
                    continue
                    
                # Create temporary file
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                    temp_filename = temp_file.name
                    
                # Write audio to file
                with wave.open(temp_filename, 'wb') as wf:
                    wf.setnchannels(self.CHANNELS)
                    wf.setsampwidth(self.pyaudio_instance.get_sample_size(self.FORMAT))
                    wf.setframerate(self.RATE)
                    for chunk in audio_buffer:
                        wf.writeframes(chunk)
                
                # Process with both services
                google_results = self.process_google(temp_filename)
                pyannote_results = self.process_pyannote(temp_filename)
                
                # Debug: Log what pyannote detected
                if pyannote_results:
                    unique_speakers = set(speaker_id for _, _, speaker_id in pyannote_results)
                    self.log_message(f"🔍 pyannote detected {len(unique_speakers)} unique speakers: {sorted(unique_speakers)}")
                else:
                    self.log_message("⚠️ pyannote detected no speakers - all will be assigned to Speaker 0")
                
                # Combine and display results
                self.log_message(f"🔍 Processing results: Google={len(google_results)}, pyannote={len(pyannote_results)}")
                
                if google_results and pyannote_results:
                    # Both Google and pyannote have results - combine them
                    self.log_message("✅ Both Google and pyannote have results - combining")
                    combined_results = self.combine_results(google_results, pyannote_results)
                    self.display_results(combined_results)
                elif google_results and not pyannote_results:
                    # Only Google has results - display with default speaker
                    self.log_message("✅ Only Google has results - displaying with default speaker")
                    self.display_google_only_results(google_results)
                elif pyannote_results and not google_results:
                    # Only pyannote has results - show speaker timeline without transcription
                    self.log_message("✅ Only pyannote has results - displaying speaker activity")
                    self.display_pyannote_only_live_results(pyannote_results)
                else:
                    # Neither has results
                    self.log_message("⚠️ Neither Google nor pyannote has results - nothing to display")
                
                # Clean up
                os.unlink(temp_filename)
                audio_buffer = []
                
            except Exception as e:
                self.log_message(f"❌ Processing error: {e}")
                
    def process_google(self, audio_file):
        """Process with Google Speech-to-Text"""
        try:
            # Detect WAV file sample rate
            sample_rate = self.RATE  # default
            try:
                with wave.open(audio_file, 'rb') as wav_file:
                    sample_rate = wav_file.getframerate()
                    self.log_message(f"🔍 Detected sample rate: {sample_rate} Hz")
            except Exception as e:
                self.log_message(f"⚠️ Could not detect sample rate, using default {self.RATE} Hz: {e}")
            
            with open(audio_file, "rb") as f:
                audio_content = f.read()
                
            audio = speech.RecognitionAudio(content=audio_content)
            config = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=sample_rate,  # Use detected rate
                language_code=self.language_var.get(),
                enable_automatic_punctuation=True,
                enable_word_time_offsets=True,
            )
            
            response = self.speech_client.recognize(config=config, audio=audio)
            
            results = []
            for result in response.results:
                alternative = result.alternatives[0]
                
                # Extract word timings
                words = []
                for word_info in alternative.words:
                    start_time = word_info.start_time.total_seconds()
                    end_time = word_info.end_time.total_seconds()
                    words.append({
                        'word': word_info.word,
                        'start_time': start_time,
                        'end_time': end_time
                    })
                
                results.append({
                    'transcript': alternative.transcript,
                    'confidence': alternative.confidence,
                    'words': words
                })
                
                # Update Google display
                timestamp = time.strftime("%H:%M:%S")
                self.google_text.insert(tk.END, 
                    f"[{timestamp}] {alternative.transcript} (conf: {alternative.confidence:.2f})\n")
                self.google_text.see(tk.END)
                
            self.log_message(f"🗣️ Google: {len(results)} transcription results")
            return results
            
        except Exception as e:
            self.log_message(f"❌ Google processing error: {e}")
            return []
            
    def process_pyannote(self, audio_file):
        """Process with pyannote speaker diarization"""
        try:
            diarization = self.diarization_pipeline(audio_file)
            
            results = []
            local_speaker_mapping = {}  # For this segment only
            
            # First pass: create local mapping for this segment
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                if speaker not in local_speaker_mapping:
                    local_speaker_mapping[speaker] = len(local_speaker_mapping)
            
            # Second pass: map to global consistent speaker IDs
            global_results = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                start_time = turn.start
                end_time = turn.end
                local_id = local_speaker_mapping[speaker]
                
                # Map to global consistent speaker ID
                global_speaker_id = self.get_global_speaker_id(speaker, start_time, end_time, audio_file)
                
                results.append((start_time, end_time, local_id))
                global_results.append((start_time, end_time, global_speaker_id))
                
                # Update pyannote display with both local and global IDs
                timestamp = time.strftime("%H:%M:%S")
                self.pyannote_text.insert(tk.END, 
                    f"[{timestamp}] Local Speaker {local_id} → Global Speaker {global_speaker_id} ({start_time:.1f}s-{end_time:.1f}s) [pyannote: {speaker}]\n")
                self.pyannote_text.see(tk.END)
            
            # Use global results for speaker assignment
            results = global_results
            unique_speakers = set(speaker_id for _, _, speaker_id in results)
            
            self.log_message(f"👤 pyannote: {len(results)} speaker segments found")
            self.log_message(f"🔍 Local speakers in this segment: {len(local_speaker_mapping)} → Global speakers: {len(unique_speakers)}")
            if local_speaker_mapping:
                self.log_message(f"🔍 Segment mapping: {local_speaker_mapping}")
            
            return results
            
        except Exception as e:
            self.log_message(f"❌ pyannote processing error: {e}")
            return []
    
    def get_global_speaker_id(self, pyannote_speaker, start_time, end_time, audio_file):
        """Map pyannote speaker to global consistent speaker ID"""
        
        # If we've seen this exact pyannote speaker label before, use the same global ID
        if pyannote_speaker in self.global_speaker_mapping:
            return self.global_speaker_mapping[pyannote_speaker]
        
        # For new speakers, try to match with existing speakers based on timing and voice characteristics
        # For now, assign new global ID (can be enhanced with voice similarity matching)
        global_id = self.next_global_speaker_id
        self.global_speaker_mapping[pyannote_speaker] = global_id
        self.next_global_speaker_id += 1
        
        self.log_message(f"🆕 New global speaker {global_id} assigned to pyannote {pyannote_speaker}")
        
        return global_id
            
    def combine_results(self, google_results, pyannote_results):
        """Combine Google transcripts with pyannote speaker labels"""
        combined = []
        
        for google_result in google_results:
            for word_data in google_result['words']:
                word_start = word_data['start_time']
                word_end = word_data['end_time']
                word_mid = (word_start + word_end) / 2
                
                # Find best matching speaker segment
                best_speaker = 0  # default to Speaker 0
                best_overlap = 0
                
                for segment_start, segment_end, speaker_id in pyannote_results:
                    # Calculate overlap
                    overlap_start = max(word_start, segment_start)
                    overlap_end = min(word_end, segment_end)
                    
                    if overlap_end > overlap_start:
                        overlap = overlap_end - overlap_start
                        if overlap > best_overlap:
                            best_overlap = overlap
                            best_speaker = speaker_id
                
                # If no speaker overlap found, use temporal proximity
                if best_overlap == 0 and pyannote_results:
                    closest_distance = float('inf')
                    for segment_start, segment_end, speaker_id in pyannote_results:
                        # Distance from word to segment
                        if word_mid < segment_start:
                            distance = segment_start - word_mid
                        elif word_mid > segment_end:
                            distance = word_mid - segment_end
                        else:
                            distance = 0  # word is within segment
                            
                        if distance < closest_distance:
                            closest_distance = distance
                            best_speaker = speaker_id
                
                combined.append({
                    'word': word_data['word'],
                    'start_time': word_start,
                    'end_time': word_end,
                    'speaker': best_speaker,
                    'confidence': google_result['confidence']
                })
                
        return combined
        
    def display_results(self, combined_results):
        """Display combined results"""
        if not combined_results:
            return
            
        # Group by speaker for conversation view
        conversations = []
        current_speaker = None
        current_text = ""
        current_start = None
        
        for word_data in combined_results:
            if current_speaker != word_data['speaker']:
                if current_text.strip():
                    conversations.append({
                        'speaker': current_speaker,
                        'text': current_text.strip(),
                        'start_time': current_start
                    })
                current_speaker = word_data['speaker']
                current_text = ""
                current_start = word_data['start_time']
                
            current_text += word_data['word'] + " "
            
        # Add last conversation
        if current_text.strip():
            conversations.append({
                'speaker': current_speaker,
                'text': current_text.strip(),
                'start_time': current_start
            })
            
        # Display conversations with color coding
        speaker_colors = {0: "#e74c3c", 1: "#3498db", 2: "#2ecc71", 3: "#f39c12", 4: "#9b59b6", 5: "#1abc9c"}
        
        for conv in conversations:
            speaker_id = conv['speaker']
            color = speaker_colors.get(speaker_id, "#333333")
            
            # Update speaker statistics
            words = len(conv['text'].split())
            if speaker_id not in self.speaker_stats:
                self.speaker_stats[speaker_id] = 0
            self.speaker_stats[speaker_id] += words
            
            timestamp = time.strftime("%H:%M:%S")
            self.transcript_text.insert(tk.END, 
                f"[{timestamp}] Speaker {speaker_id}: {conv['text']}\n")
            
            # Apply color to speaker text
            start_index = self.transcript_text.index("end-2l")
            end_index = self.transcript_text.index("end-1l")
            self.transcript_text.tag_add(f"speaker_{speaker_id}", start_index, end_index)
            self.transcript_text.tag_config(f"speaker_{speaker_id}", foreground=color)
            
        self.transcript_text.see(tk.END)
        
        # Update speaker statistics
        total_words = sum(self.speaker_stats.values())
        stats_text = "👥 Speaker Statistics: "
        for speaker_id, word_count in self.speaker_stats.items():
            percentage = (word_count / total_words * 100) if total_words > 0 else 0
            stats_text += f"Speaker {speaker_id}: {word_count} words ({percentage:.1f}%) | "
        
        self.stats_label.config(text=stats_text.rstrip(" | "))
        
    def clear_all(self):
        """Clear all displays"""
        self.transcript_text.delete(1.0, tk.END)
        self.google_text.delete(1.0, tk.END)
        self.pyannote_text.delete(1.0, tk.END)
        self.speaker_stats.clear()
        self.stats_label.config(text="👥 Speaker Statistics: Waiting for speech...")
        self.log_message("🗑️ All displays cleared")
        
    def run(self):
        """Start the application"""
        self.log_message("🚀 Authenticated Hybrid Speech + Speaker Diarization Started")
        self.log_message("🔐 All authentication checks completed")
        self.log_message("🎯 Combining Google Speech Recognition with pyannote Speaker Diarization")
        self.log_message("💡 This provides the best accuracy for both speech and speakers")
        self.root.mainloop()

def main():
    print("🔐 Authenticated Hybrid Speech Recognition + Speaker Diarization")
    print("=" * 60)
    print()
    
    # Check authentication status
    print("🔍 Checking authentication status...")
    
    # Check Google credentials
    if os.path.exists("voice-sun-1-67f7efc777f3.json"):
        print("✅ Google Cloud credentials found")
    else:
        print("❌ Google Cloud credentials not found")
        
    # Check Hugging Face authentication
    try:
        from huggingface_hub import HfFolder
        token = HfFolder.get_token()
        if token:
            print("✅ Hugging Face authentication found")
        else:
            print("❌ Hugging Face not authenticated")
            print("💡 Run: huggingface-cli login")
    except:
        print("❌ Hugging Face authentication check failed")
        
    # Check pyannote availability
    if PYANNOTE_AVAILABLE:
        print("✅ pyannote-audio available")
    else:
        print("❌ pyannote-audio not available")
        
    print()
    print("🚀 Starting application...")
    
    app = AuthenticatedHybridDiarization()
    app.run()

if __name__ == "__main__":
    main()

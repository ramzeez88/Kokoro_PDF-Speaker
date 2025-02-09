import tkinter as tk
from tkinter import ttk, messagebox, filedialog, scrolledtext
import fitz  # PyMuPDF
import docx
from kokoro import KPipeline
import sounddevice as sd
import threading
import time
import gc
import torch


class CombinedAppGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Kokoro_PDF-Speaker")
        self.root.geometry("550x630")
        self.center_window(30)

        # --- Kokoro TTS Section ---
        self.kokoro_frame = ttk.LabelFrame(self.root, text="Select device:")
        self.kokoro_frame.pack(fill="x", padx=10, pady=5)

        self.device_var = tk.StringVar(value='cpu')
        self.device_var.trace_add("write", self.on_device_change)
        self.device_frame = ttk.Frame(self.kokoro_frame)
        self.device_frame.pack(fill="x", padx=10, pady=5)
        ttk.Radiobutton(self.device_frame, text="CPU", variable=self.device_var, value='cpu').pack(side="left",
                                                                                                   padx=5, pady=5)
        ttk.Radiobutton(self.device_frame, text="CUDA", variable=self.device_var, value='cuda').pack(side="left",
                                                                                                    padx=5, pady=5)

        self.voice_labelframe = ttk.LabelFrame(self.kokoro_frame, text="Voice Selection")
        self.voice_labelframe.pack(fill="x", padx=10, pady=5)

        self.voice_var = tk.StringVar(value='af_heart')
        self.voice_frame = ttk.Frame(self.voice_labelframe)
        self.voice_frame.pack(fill="x", padx=10, pady=5)
        ttk.Label(self.voice_frame, text="Select voice:").pack(side="left", padx=5, pady=5)
        self.voice_dropdown = ttk.Combobox(self.voice_frame, textvariable=self.voice_var, state="readonly")
        self.voice_dropdown['values'] = ('af_heart', 'af_bella', 'af_jessica', 'af_sarah', 'am_adam', 'am_michael',
                                          'bf_emma', 'bf_isabella', 'bm_george', 'bm_lewis', 'af_nicole', 'af_sky')
        self.voice_dropdown.pack(fill="x", padx=5, pady=5)
        self.current_voice = None
        self.current_speed = None

        self.auto_advance_var = tk.BooleanVar(value=True)
        self.auto_advance_frame = ttk.Frame(self.kokoro_frame)
        self.auto_advance_frame.pack(fill="x", padx=10, pady=5)
        self.auto_advance_check = ttk.Checkbutton(self.auto_advance_frame,
                                                text="Auto-advance to next page",
                                                variable=self.auto_advance_var)
        self.auto_advance_check.pack(side="left", padx=5, pady=5)

        self.speed_var = tk.DoubleVar(value=1.0)
        self.speed_frame = ttk.Frame(self.kokoro_frame)
        self.speed_frame.pack(fill="x", padx=10, pady=5)

        self.speed_scale = ttk.Scale(
            self.speed_frame,
            from_=0.5,
            to=2.0,
            variable=self.speed_var,
            orient="horizontal",
            command=self.update_speed_label
        )
        self.speed_scale.pack(side="left", fill="x", expand=True, padx=5, pady=5)

        self.speed_label_var = tk.StringVar()
        self.speed_label = ttk.Label(self.speed_frame, textvariable=self.speed_label_var)
        self.speed_label.pack(side="left", padx=5)
        self.update_speed_label(None)

        # --- Text Extraction Section ---
        self.extractor_frame = ttk.LabelFrame(self.root, text="Text Extractor")
        self.extractor_frame.pack(fill="both", expand=True, padx=10, pady=5)

        self.go_to_page_frame = ttk.Frame(self.extractor_frame)
        self.go_to_page_frame.grid(row=2, column=0, columnspan=3, padx=5, pady=5, sticky="ew")

        self.left_button = ttk.Button(self.go_to_page_frame, text="←", width=3, command=self.previous_page)
        self.left_button.pack(side="left", padx=5)

        self.right_button = ttk.Button(self.go_to_page_frame, text="→", width=3, command=self.next_page)
        self.right_button.pack(side="left", padx=5)

        self.go_to_page_label = ttk.Label(self.go_to_page_frame, text="Go to Page:")
        self.go_to_page_label.pack(side="left", padx=5)
        self.page_number_entry = ttk.Entry(self.go_to_page_frame, width=10)
        self.page_number_entry.pack(side="left", padx=5)
        self.go_to_page_button = ttk.Button(self.go_to_page_frame, text="Go", command=self.go_to_page)
        self.go_to_page_button.pack(side="left", padx=5)

        self.file_path_label = tk.Label(self.extractor_frame, text="File Path:")
        self.file_path_label.grid(row=0, column=0, padx=5, pady=5, sticky="e")
        self.file_path_entry = tk.Entry(self.extractor_frame, width=50)
        self.file_path_entry.grid(row=0, column=1, padx=5, pady=5)
        self.browse_button = tk.Button(self.extractor_frame, text="Browse", command=self.browse_file)
        self.browse_button.grid(row=0, column=2, padx=5, pady=5)

        # --- Single ScrolledText for Display ---
        self.text_area = scrolledtext.ScrolledText(self.extractor_frame, wrap=tk.WORD, width=60, height=10)
        self.text_area.grid(row=1, column=0, columnspan=3, padx=10, pady=10, sticky="nsew")
        self.text_area.configure(state='disabled')

        self.extractor_frame.grid_rowconfigure(1, weight=1)
        self.extractor_frame.grid_columnconfigure(0, weight=1)
        self.extractor_frame.grid_columnconfigure(1, weight=1)
        self.extractor_frame.grid_columnconfigure(2, weight=1)

        # --- Buttons (Combined) ---
        self.button_frame = ttk.Frame(self.root)
        self.button_frame.pack(fill="x", padx=10, pady=5)

        self.play_button = ttk.Button(self.button_frame, text="Play", command=self.play_audio)
        self.play_button.pack(side="left", padx=5, pady=5)
        self.pause_button = ttk.Button(self.button_frame, text="Pause/Resume", command=self.pause_resume_audio)
        self.pause_button.pack(side="left", padx=5, pady=5)
        self.stop_button = ttk.Button(self.button_frame, text="Stop", command=self.stop_audio)
        self.stop_button.pack(side="left", padx=5, pady=5)

        self.copy_button = ttk.Button(self.button_frame, text="Copy to Clipboard", command=self.copy_to_clipboard)
        self.copy_button.pack(side="left", padx=5, pady=5)

        self.exit_button = ttk.Button(self.button_frame, text="Exit", command=self.on_exit)
        self.exit_button.pack(side="right", padx=5, pady=5)

        # --- Audio Control Variables ---
        self.audio_thread = None
        self.is_playing = False
        self.is_paused = False
        self.stop_playback = False
        self.audio_data = None
        self.current_position = 0
        self.start_timestamp = 0
        self.accumulated_time = 0
        self.stream = None
        self.pipeline = None

        # --- Page Management ---
        self.current_page = 1
        self.pages = []  # List to store the text of each page
        self.total_pages = 0
        self.is_paginated = False # Flag to indicate if the document is paginated

        self.initialize_pipeline()

    def center_window(self, y_offset=0):
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        x = (screen_width // 2) - (width // 2)
        y = y_offset
        self.root.geometry(f"{width}x{height}+{x}+{y}")

    def previous_page(self):
        if self.current_page > 1:
            self.current_page -= 1
            self.display_page()

    def next_page(self):
        if self.current_page < self.total_pages:
            self.current_page += 1
            self.display_page()

    def go_to_page(self):
        """Navigates to the specified page number."""
        page_number_str = self.page_number_entry.get()
        if not page_number_str:
            messagebox.showwarning("Warning", "Please enter a page number.")
            return

        try:
            page_number = int(page_number_str)
            if page_number <= 0 or page_number > self.total_pages:
                raise ValueError
            self.current_page = page_number
            self.display_page()
        except ValueError:
            messagebox.showerror("Error", "Invalid page number. Please enter a valid page number.")

    def display_page(self):
        """Displays the content of the current page in the text area."""
        if 1 <= self.current_page <= self.total_pages:
            self.text_area.configure(state='normal')
            self.text_area.delete("1.0", tk.END)
            if self.is_paginated:
                page_text = self.pages[self.current_page - 1]
                if self.current_page == self.total_pages:
                    page_text += "\nEnd of Document"
                self.text_area.insert(tk.END, f"Page {self.current_page},\n{page_text}")
            else:
                page_text = self.pages[0]
                if self.total_pages > 0:
                  page_text += "\nEnd of Document"
                self.text_area.insert(tk.END, page_text)
            self.text_area.configure(state='disabled')
            self.text_area.see("1.0")

    def update_speed_label(self, value):
        current_speed = self.speed_var.get()
        self.speed_label_var.set(f"Speed: {current_speed:.2f}x")

    def initialize_pipeline(self):
        self.clear_pipeline()
        try:
            self.pipeline = KPipeline(lang_code='a', device=self.device_var.get())
        except RuntimeError as e:
            if "No CUDA GPUs are available" in str(e):
                messagebox.showerror("CUDA Error", "No CUDA GPUs are available.  Switching to CPU.")
                self.device_var.set('cpu')
                self.pipeline = KPipeline(lang_code='a', device='cpu')
            else:
                messagebox.showerror("Initialization Error", str(e))
        except Exception as e:
            messagebox.showerror("Initialization Error", str(e))

    def on_device_change(self, *args):
        self.initialize_pipeline()

    def play_audio(self):
        if self.is_playing:
            messagebox.showwarning("Warning", "Audio is already playing.")
            return

        if not self.pages:
            messagebox.showwarning("Warning", "Please load a document first.")
            return

        text = self.text_area.get("1.0", "end-1c").strip()
        if not text:
            messagebox.showwarning("Warning", "No text to speak on this page.")
            return

        MAX_CHUNK_SIZE = 1000
        chunks = self.split_text_into_chunks(text, MAX_CHUNK_SIZE)

        self.play_button.config(state="disabled")
        self.is_playing = True
        self.is_paused = False
        self.stop_playback = False

        device = self.device_var.get()
        voice = self.voice_var.get()
        speed = self.speed_var.get()

        self.current_voice = voice
        self.current_speed = speed

        self.audio_thread = threading.Thread(target=self.generate_and_play_audio,
                                        args=(chunks, voice, speed))
        self.audio_thread.start()

    def split_text_into_chunks(self, text, max_size):
        sentences = text.replace('\n', ' ').split('. ')
        chunks = []
        current_chunk = []
        current_size = 0

        for sentence in sentences:
            if sentence != sentences[-1]:
                sentence += '.'

            sentence_size = len(sentence)
            if current_size + sentence_size > max_size and current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]
                current_size = sentence_size
            else:
                current_chunk.append(sentence)
                current_size += sentence_size

        if current_chunk:
            chunks.append(' '.join(current_chunk))

        return chunks

    def generate_and_play_audio(self, text_chunks, voice, speed):
        self.audio_data = []
        try:
            self.root.after(0, lambda: self.play_button.config(state="disabled"))

            for chunk in text_chunks:
                if self.stop_playback:
                    break

                generator = self.pipeline(chunk, voice=voice, speed=speed, split_pattern=r'\n+')
                for i, (gs, ps, audio) in enumerate(generator):
                    if self.stop_playback:
                        break
                    self.audio_data.extend(audio)

            if not self.stop_playback and self.audio_data:
                self.audio_data = [float(x) for x in self.audio_data]
                self.current_position = 0
                self.accumulated_time = 0
                self.start_timestamp = time.time()

                self.stream = sd.OutputStream(
                    samplerate=24000,
                    channels=1,
                    callback=self.audio_callback,
                    blocksize=1024
                )
                self.stream.start()

                while self.current_position < len(self.audio_data) and not self.stop_playback:
                    sd.sleep(10)

                if self.stream:
                    self.stream.stop()
                    self.stream.close()
                    self.stream = None

                # Auto-advance logic (using root.after)
                if not self.stop_playback and self.auto_advance_var.get():
                    if self.current_page < self.total_pages:
                        self.next_page()
                        # Schedule play_audio for the next page after a delay
                        self.root.after(50, self.play_audio)  # 50ms delay
                        return

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")
        finally:
            if not self.stop_playback and not self.is_paused:
                self.is_playing = False
                # No need to check for auto-advance here, it's handled above
                self.root.after(0, lambda: self.play_button.config(state="normal"))

    def audio_callback(self, outdata, frames, time_info, status, userdata=None):
        if status:
            print(status)
        if self.is_paused or self.stop_playback:
            outdata[:] = 0
            return

        chunksize = min(len(self.audio_data) - self.current_position, frames)
        outdata[:chunksize, 0] = self.audio_data[self.current_position:self.current_position + chunksize]
        if chunksize < frames:
            outdata[chunksize:, 0] = 0
        self.current_position += chunksize

    def pause_resume_audio(self):
        if not self.is_playing and not self.is_paused:
            messagebox.showwarning("Warning", "No audio is currently playing.")
            return

        if self.is_paused:
            new_voice = self.voice_var.get()
            speed = self.speed_var.get()

            if new_voice != self.current_voice or speed != self.current_speed:
                text = self.text_area.get("1.0", "end-1c").strip()
                words = text.split()
                total_chars = len(text)
                approx_position = int(total_chars * (self.current_position / len(self.audio_data)))

                word_position = 0
                char_count = 0
                for i, word in enumerate(words):
                    if char_count + len(word) > approx_position:
                        word_position = i - 1
                        break
                    char_count += len(word) + 1

                remaining_text = ' '.join(words[max(0, word_position):])

                self.stop_audio()

                self.stop_playback = False
                self.is_paused = False
                self.is_playing = True
                self.current_voice = new_voice
                self.current_speed = speed
                chunks = self.split_text_into_chunks(remaining_text, 1000)
                self.audio_thread = threading.Thread(target=self.generate_and_play_audio,
                                                args=(chunks, new_voice, speed))
                self.audio_thread.start()
                self.pause_button.config(text="Pause")
                return

            self.is_paused = False
            self.start_timestamp = time.time() - self.accumulated_time
            self.pause_button.config(text="Pause")
        else:
            self.is_paused = True
            self.accumulated_time = time.time() - self.start_timestamp
            self.pause_button.config(text="Resume")

    def stop_audio(self):
        self.stop_playback = True
        if self.audio_thread and self.audio_thread.is_alive():
            self.audio_thread.join(timeout=1)
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        self.is_playing = False
        self.is_paused = False
        self.current_position = 0
        self.accumulated_time = 0
        self.pause_button.config(text="Pause/Resume")
        self.play_button.config(state="normal")
        if hasattr(self, 'next_audio_thread'):
            self.next_audio_thread = None

    def clear_pipeline(self):
        if self.pipeline:
            del self.pipeline
            self.pipeline = None
            if self.device_var.get() == 'cuda':
                torch.cuda.empty_cache()
            gc.collect()

    def browse_file(self):
        filepath = filedialog.askopenfilename(
            filetypes=[
                ("PDF Files", "*.pdf"),
                ("Text Files", "*.txt"),
                ("Word Documents", "*.docx"),
                ("All Files", "*.*"),
            ]
        )
        if filepath:
            self.file_path_entry.delete(0, tk.END)
            self.file_path_entry.insert(0, filepath)
            self.extract_and_display(filepath)

    def extract_and_display(self, filepath):
        file_ext = filepath.lower().split('.')[-1]

        if file_ext == "pdf":
            self.pages = self.extract_text_from_pdf_per_page(filepath)
            self.is_paginated = True
        elif file_ext == "txt":
            self.pages = self.extract_text_from_txt(filepath)
            self.is_paginated = False
        elif file_ext == "docx":
            self.pages = self.extract_text_from_docx(filepath)
            self.is_paginated = False
        else:
            self.pages = []
            self.is_paginated = False

        self.total_pages = len(self.pages) if self.pages else 0
        self.current_page = 1
        self.display_page()

    def extract_text_from_pdf_per_page(self, pdf_path):
        try:
            with fitz.open(pdf_path) as doc:
                text_list = []
                for page_num, page in enumerate(doc):
                    page_text = page.get_text()
                    text_list.append(page_text)
                return text_list
        except Exception as e:
            print(f"An error occurred (PDF): {e}")
            return []

    def extract_text_from_txt(self, txt_path):
        try:
            with open(txt_path, "r", encoding="utf-8") as f:
                text = f.read()
                return [text]
        except Exception as e:
            print(f"An error occurred (TXT): {e}")
            return []
        except UnicodeDecodeError:
            print(f"UnicodeDecodeError with UTF-8. Trying latin-1.")
            try:
                with open(txt_path, 'r', encoding='latin-1') as f:
                    text = f.read()
                    return [text]
            except Exception as e:
                print(f"An error occurred (TXT, latin-1): {e}")
                return []

    def extract_text_from_docx(self, docx_path):
        try:
            doc = docx.Document(docx_path)
            full_text = []
            for paragraph in doc.paragraphs:
                full_text.append(paragraph.text)
            extracted_text = "\n".join(full_text)

            if not extracted_text.strip():
                print("DOCX file is empty.")
                return []

            return [extracted_text]

        except docx.opc.exceptions.PackageNotFoundError:
            print("PackageNotFoundError: The file may not be a valid .docx file.")
            return []
        except Exception as e:
            print(f"An error occurred (DOCX): {e}")
            print(f"Exception Type: {type(e)}")
            print(f"Exception Message: {e}")
            return []

    def copy_to_clipboard(self):
        try:
            text = self.text_area.get("1.0", "end-1c")
            self.root.clipboard_clear()
            self.root.clipboard_append(text)
            messagebox.showinfo("Copied", "Text copied to clipboard!")
        except Exception as e:
            messagebox.showerror("Error", f"Could not copy text: {e}")

    def on_exit(self):
        self.stop_audio()
        self.clear_pipeline()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = CombinedAppGUI(root)
    root.mainloop()

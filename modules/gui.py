import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk, ImageDraw
import threading


class AppGUI:
    def __init__(self, master, classify_callback):
        self.master = master
        self.master.title("Online Gambling Ad Classification")
        self.master.geometry("960x540")
        self.master.resizable(False, False)
        self.classify_callback = classify_callback
        self.image_path = None
        self.image_label_img = None
        self.cache = {}  # Initialize cache to store temporary results

        # === Main Frame ===
        main_frame = tk.Frame(master, bg="#fff")
        main_frame.pack(fill="both", expand=True, padx=8, pady=8)
        main_frame.grid_rowconfigure(0, weight=1)
        main_frame.grid_columnconfigure(0, weight=1)
        main_frame.grid_columnconfigure(1, weight=1)

        # === LEFT ===
        left_frame = tk.Frame(main_frame, bg="#fff")
        left_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 0), pady=0)
        left_frame.grid_rowconfigure(3, weight=1)
        left_frame.grid_columnconfigure(0, weight=1)

        # Static frame for image and buttons (so they don't shift)
        static_frame = tk.Frame(left_frame, bg="#fff")
        static_frame.grid(row=0, column=0, sticky="n", pady=(18, 0))

        # Image Preview (original size)
        self.image_label = tk.Label(static_frame, width=320, height=220, bg="#f8f9fa", relief="groove", borderwidth=2)
        self.image_label.pack(pady=(0, 18))

        # Placeholder image
        placeholder = Image.new("RGB", (320, 220), "#f8f9fa")
        draw = ImageDraw.Draw(placeholder)
        draw.rectangle([0, 0, 319, 219], outline="#bbb")
        self.image_label_img = ImageTk.PhotoImage(placeholder)
        self.image_label.config(image=self.image_label_img)

        # Buttons
        btn_frame = tk.Frame(static_frame, bg="#fff")
        btn_frame.pack(pady=(0, 10))
        style = {"font": ("Segoe UI", 11, "bold"), "bg": "#fff", "relief": "groove", "borderwidth": 2, "cursor": "hand2", "activebackground": "#e9ecef"}
        self.upload_btn = tk.Button(btn_frame, text="Upload Ad Image", command=self.upload_image, width=22, height=2, **style)
        self.upload_btn.pack(side="left", padx=(0, 14))
        self.classify_btn = tk.Button(btn_frame, text="Classify", command=self.on_classify, width=22, height=2, **style)
        self.classify_btn.pack(side="left")

        # --- Classification and OCR result area ---
        result_area = tk.Frame(left_frame, bg="#fff")
        result_area.grid(row=1, column=0, sticky="ew", pady=(10, 0))

        # Center the container for the boxes
        box_container = tk.Frame(result_area, bg="#fff")
        box_container.pack()

        # Classification Result
        self.result_box_frame = tk.Frame(box_container, bg="#fff")
        self.result_box_frame.pack(side="left", padx=(0, 8))
        tk.Label(self.result_box_frame, text="Classification Result", font=("Segoe UI", 13, "bold"), bg="#fff", fg="#222").pack(anchor="center", pady=(0, 8))
        self.result_box = tk.Label(self.result_box_frame, text="-", font=("Segoe UI", 12), bg="#f8f9fa", fg="#d90429", relief="groove", borderwidth=2, width=28, height=3, anchor="center", justify="center", padx=8, pady=8)
        self.result_box.pack(padx=4, pady=(0, 8))

        # OCR Result (setup but not packed)
        self.ocr_box_frame = tk.Frame(box_container, bg="#fff")
        self.ocr_label = tk.Label(self.ocr_box_frame, text="Text Extraction Result (OCR)", font=("Segoe UI", 13, "bold"), bg="#fff", fg="#222")
        self.ocr_label.pack(anchor="center", pady=(0, 8))
        self.ocr_text_box = tk.Text(self.ocr_box_frame, height=5, width=38, wrap="word", font=("Segoe UI", 11), bg="#f8f9fa", relief="groove", borderwidth=2, padx=8, pady=8)
        self.ocr_text_box.pack(padx=4, pady=(0, 8))
        self.ocr_text_box.insert("1.0", "")
        self.ocr_text_box.config(state="disabled")

        # === RIGHT ===
        right_frame = tk.Frame(main_frame, bg="#fff")
        right_frame.grid(row=0, column=1, sticky="n", padx=(10, 0), pady=0)
        right_frame.grid_rowconfigure(99, weight=1)

        # Classification Method
        tk.Label(right_frame, text="Classification Method", font=("Segoe UI", 11, "bold"), bg="#fff").pack(pady=(40, 5))
        self.mode_var = tk.StringVar(value="CNN")
        mode_frame = tk.Frame(right_frame, bg="#fff")
        mode_frame.pack(pady=(0, 18))
        self.mode_combo = ttk.Combobox(
            mode_frame, textvariable=self.mode_var, state="readonly",
            values=["CNN", "OCR BERT", "CNN & OCR BERT"], width=20, font=("Segoe UI", 10)
        )
        self.mode_combo.pack(ipady=4)
        self.mode_combo.bind("<<ComboboxSelected>>", lambda e: self.update_visibility())

        # CNN Model group (label + combobox)
        self.cnn_group = tk.Frame(right_frame, bg="#fff")
        self.cnn_label = tk.Label(self.cnn_group, text="CNN Model", font=("Segoe UI", 11, "bold"), bg="#fff")
        self.cnn_label.pack(anchor="w", pady=(0, 5))
        self.cnn_model_var = tk.StringVar()
        self.cnn_model_combo = ttk.Combobox(
            self.cnn_group, textvariable=self.cnn_model_var, state="readonly",
            values=["EfficientNet-B0", "ResNet-50"], width=20, font=("Segoe UI", 10)
        )
        self.cnn_model_combo.pack(ipady=4)
        self.cnn_group.pack(pady=(0, 18))
        self.cnn_model_combo.current(0)  # Set default to EfficientNet

        # OCR Model group (label + combobox)
        self.ocr_group_model = tk.Frame(right_frame, bg="#fff")
        self.ocr_label_model = tk.Label(self.ocr_group_model, text="Text Extraction Model (OCR)", font=("Segoe UI", 11, "bold"), bg="#fff")
        self.ocr_label_model.pack(anchor="w", pady=(0, 5))
        self.ocr_engine_var = tk.StringVar(value="PaddleOCR")
        self.ocr_engine_combo = ttk.Combobox(
            self.ocr_group_model, textvariable=self.ocr_engine_var, state="readonly",
            values=["PaddleOCR", "EasyOCR"], width=20, font=("Segoe UI", 10)
        )
        self.ocr_engine_combo.pack(ipady=4)
        self.ocr_group_model.pack(pady=(0, 18))

        self.update_visibility()
        self.master.update()
        self.master.deiconify()

    def update_visibility(self):
        # Hide all model groups first
        self.cnn_group.pack_forget()
        self.ocr_group_model.pack_forget()

        mode = self.mode_var.get()

        # Manage visibility of CNN model selection
        if mode in ("CNN", "CNN & OCR BERT"):
            self.cnn_group.pack(pady=(0, 18))

        # Manage visibility of OCR model selection and OCR result box
        if mode in ("OCR BERT", "CNN & OCR BERT"):
            self.ocr_group_model.pack(pady=(0, 18))
            self.ocr_box_frame.pack(side="left", padx=(8, 0))
        else:
            # Hide OCR box if not in a mode that uses it
            self.ocr_box_frame.pack_forget()

    def upload_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png *.jpg *.jpeg *.webp")])
        if file_path:
            # If the image path changes, reset the cache
            if self.image_path != file_path:
                self.cache = {}
                print("New image uploaded, cache cleared.")

            self.image_path = file_path
            
            # Open the image and convert to RGB
            img = Image.open(file_path).convert("RGB")

            # If the file is webp, convert and save as a temporary jpg
            if file_path.lower().endswith(".webp"):
                import tempfile
                import os
                
                # Create a temporary file to save the jpg version
                fd, temp_jpg_path = tempfile.mkstemp(suffix=".jpg")
                os.close(fd) # Close the file descriptor as Pillow will handle it
                
                img.save(temp_jpg_path, 'jpeg')
                print(f"WebP image converted to temporary JPG at: {temp_jpg_path}")
                
                # Use the temporary jpg path for processing
                self.image_path = temp_jpg_path

            # Display thumbnail in the GUI
            img.thumbnail((300, 200))
            self.image_label_img = ImageTk.PhotoImage(img)
            self.image_label.config(image=self.image_label_img)

    def on_classify(self):
        if not self.image_path:
            messagebox.showwarning("Warning", "Please upload an image first.")
            return
        self.show_loading()
        self.ocr_text_box.config(state="normal")
        self.ocr_text_box.delete("1.0", "end")
        self.ocr_text_box.insert("1.0", "\u23F3 Processing text (if any)...")
        self.ocr_text_box.config(state="disabled")
        thread = threading.Thread(target=self.run_classification)
        thread.start()

    def run_classification(self):
        # Mapping value for backend
        mode_map = {
            "CNN": "cnn",
            "OCR BERT": "ocr_bert",
            "CNN & OCR BERT": "both"
        }
        cnn_map = {
            "ResNet-50": "resnet",
            "EfficientNet-B0": "efficientnet"
        }
        ocr_map = {
            "PaddleOCR": "paddleocr",
            "EasyOCR": "easyocr"
        }
        self.classify_callback(
            self.image_path,
            mode_map[self.mode_var.get()],
            cnn_map.get(self.cnn_model_var.get(), "resnet"),
            ocr_map.get(self.ocr_engine_var.get(), "paddleocr"),
            show_result=self.show_result,
            show_text=self.show_ocr_text,
            cache=self.cache  # Send cache to the backend function
        )

    def show_loading(self):
        self.result_box.config(text="\u23F3 Processing...", fg="blue")

    def show_result(self, result_text, is_judi):
        color = "#d90429" if is_judi else "#2b9348"
        self.result_box.config(text=result_text, fg=color)

    def show_ocr_text(self, text):
        self.ocr_text_box.config(state="normal")
        self.ocr_text_box.delete("1.0", "end")
        self.ocr_text_box.insert("1.0", text.strip() if text.strip() else "(No text detected)")
        self.ocr_text_box.config(state="disabled")
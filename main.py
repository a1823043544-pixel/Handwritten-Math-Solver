import tkinter as tk
from tkinter import filedialog, messagebox, Scale
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageTk
import tensorflow as tf
import re
import os

# Import your local modules
import config
import utils


class MathSolverApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Handwritten Math Solver")
        self.root.geometry("900x600")
        self.root.resizable(False, False)

        # --- Variables ---
        self.mode = "DRAW"  # Options: DRAW, FILE, CAMERA
        self.pen_size = 6  # Default thickness
        self.drawing = False
        self.last_point = (0, 0)
        self.camera_active = False
        self.cap = None

        # Image containers
        # 1. For Drawing: Black background, white ink (MNIST style)
        self.draw_image = Image.new("L", (600, 400), 0)
        self.draw_brush = ImageDraw.Draw(self.draw_image)

        # 2. For File/Camera: Holds the loaded OpenCV image
        self.input_image_cv = None

        # --- Load Model ---
        print("Loading Model...")
        try:
            self.model = tf.keras.models.load_model(config.MODEL_PATH)
            print("Model loaded successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Could not load model: {e}")
            self.model = None

        # --- UI Layout ---
        self.setup_ui()

    def setup_ui(self):
        # 1. Left Control Panel
        panel = tk.Frame(self.root, width=200, bg="#f0f0f0")
        panel.pack(side=tk.LEFT, fill=tk.Y)

        # Title
        tk.Label(panel, text="Math Solver", font=("Arial", 16, "bold"), bg="#f0f0f0").pack(pady=20)

        # Mode Selection
        tk.Label(panel, text="Select Mode:", bg="#f0f0f0", font=("Arial", 10)).pack(anchor="w", padx=20)

        self.btn_mode_draw = tk.Button(panel, text="✍️ Draw Board", width=15, command=lambda: self.set_mode("DRAW"))
        self.btn_mode_draw.pack(pady=5)

        self.btn_mode_file = tk.Button(panel, text="📁 Upload File", width=15, command=lambda: self.set_mode("FILE"))
        self.btn_mode_file.pack(pady=5)

        self.btn_mode_cam = tk.Button(panel, text="📷 Camera", width=15, command=lambda: self.set_mode("CAMERA"))
        self.btn_mode_cam.pack(pady=5)

        # Pen Size Slider (Only for Draw mode)
        self.slider_frame = tk.Frame(panel, bg="#f0f0f0")
        self.slider_frame.pack(pady=20)
        tk.Label(self.slider_frame, text="Pen Size:", bg="#f0f0f0").pack()
        self.scale_pen = Scale(self.slider_frame, from_=2, to=15, orient=tk.HORIZONTAL, command=self.update_pen_size)
        self.scale_pen.set(self.pen_size)
        self.scale_pen.pack()

        # Action Buttons
        tk.Button(panel, text="Clear / Reset", bg="#ffcccc", width=15, command=self.clear_input).pack(pady=20)

        self.btn_solve = tk.Button(panel, text="SOLVE (=)", bg="#ccffcc", font=("Arial", 12, "bold"), width=15,
                                   height=2, command=self.solve)
        self.btn_solve.pack(side=tk.BOTTOM, pady=30)

        # 2. Right Display Area
        right_frame = tk.Frame(self.root, bg="white")
        right_frame.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH)

        # Result Display (Top)
        self.lbl_equation = tk.Label(right_frame, text="Equation: ???", font=("Arial", 18), bg="white", fg="blue")
        self.lbl_equation.pack(pady=10)

        self.lbl_result = tk.Label(right_frame, text="Result: ---", font=("Arial", 24, "bold"), bg="white", fg="green")
        self.lbl_result.pack(pady=10)

        # Input Area (Canvas or Label)
        container = tk.Frame(right_frame, width=600, height=400, bg="gray")
        container.pack(pady=10)
        container.pack_propagate(False)  # Don't shrink

        # Widget 1: Drawing Canvas
        self.canvas = tk.Canvas(container, width=600, height=400, bg="black", cursor="cross")
        self.canvas.place(x=0, y=0)
        self.canvas.bind("<Button-1>", self.start_draw)
        self.canvas.bind("<B1-Motion>", self.draw_move)

        # Widget 2: Image/Camera Display Label (Hidden initially)
        self.lbl_display = tk.Label(container, bg="black")

        # Start in Draw mode
        self.set_mode("DRAW")

    # --- Mode Switching ---
    def set_mode(self, mode):
        self.mode = mode
        self.stop_camera()  # Always stop camera when switching
        self.clear_input()

        # Reset UI visibility
        if mode == "DRAW":
            self.canvas.place(x=0, y=0)
            self.lbl_display.place_forget()
            self.slider_frame.pack(pady=20)  # Show slider
            self.root.title("Math Solver - Draw Mode")

        elif mode == "FILE":
            self.canvas.place_forget()
            self.lbl_display.place(x=0, y=0, width=600, height=400)
            self.slider_frame.pack_forget()  # Hide slider
            self.root.title("Math Solver - File Mode")
            self.upload_file()

        elif mode == "CAMERA":
            self.canvas.place_forget()
            self.lbl_display.place(x=0, y=0, width=600, height=400)
            self.slider_frame.pack_forget()
            self.root.title("Math Solver - Camera Mode")
            self.start_camera()

    def update_pen_size(self, val):
        self.pen_size = int(val)

    # --- Drawing Logic ---
    def start_draw(self, event):
        self.drawing = True
        self.last_point = (event.x, event.y)

    def draw_move(self, event):
        if self.drawing:
            x, y = event.x, event.y
            r = self.pen_size

            # Draw on screen (Tkinter)
            self.canvas.create_oval(x - r, y - r, x + r, y + r, fill="white", outline="white")
            self.canvas.create_line(self.last_point[0], self.last_point[1], x, y, fill="white", width=r * 2,
                                    capstyle=tk.ROUND)

            # Draw on memory image (Pillow) - Essential for model
            self.draw_brush.ellipse([x - r, y - r, x + r, y + r], fill=255)
            self.draw_brush.line([self.last_point, (x, y)], fill=255, width=r * 2)

            self.last_point = (x, y)

    # --- File Logic ---
    def upload_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Images", "*.png;*.jpg;*.jpeg")])
        if file_path:
            img = cv2.imread(file_path)
            self.input_image_cv = img
            self.show_cv_image(img)

    # --- Camera Logic ---
    def start_camera(self):
        self.camera_active = True
        self.cap = cv2.VideoCapture(0)
        self.update_camera_feed()

    def stop_camera(self):
        self.camera_active = False
        if self.cap:
            self.cap.release()
            self.cap = None

    def update_camera_feed(self):
        if self.camera_active and self.cap:
            ret, frame = self.cap.read()
            if ret:
                self.input_image_cv = frame.copy()  # Save current frame
                self.show_cv_image(frame)
                self.root.after(20, self.update_camera_feed)

    def show_cv_image(self, cv_img):
        # Convert BGR to RGB
        rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_img)

        # Resize to fit 600x400
        pil_img.thumbnail((600, 400))

        self.tk_img = ImageTk.PhotoImage(pil_img)
        self.lbl_display.config(image=self.tk_img)

    # --- Clear ---
    def clear_input(self):
        # Clear drawing
        self.canvas.delete("all")
        self.draw_image = Image.new("L", (600, 400), 0)
        self.draw_brush = ImageDraw.Draw(self.draw_image)

        # Clear file/cam
        self.lbl_display.config(image="")
        self.input_image_cv = None

        # Clear text
        self.lbl_equation.config(text="Equation: ???")
        self.lbl_result.config(text="Result: ---", fg="green")

    # --- THE CORE: Solver Logic ---
    def solve(self):
        if self.model is None: return

        # --- 1. Get the Image ---
        final_img_bgr = None

        if self.mode == "DRAW":
            img_np = np.array(self.draw_image)
            if np.mean(img_np) == 0:
                self.lbl_equation.config(text="Equation: (Empty)")
                return
            final_img_bgr = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)

        else:  # FILE or CAMERA
            if self.input_image_cv is None:
                messagebox.showwarning("Warning", "No image found!")
                return
            final_img_bgr = self.input_image_cv

        # --- 2. Process Image (Using Utils) ---
        try:
            # Returns list of tuples: (image, width, height, holes)
            char_data_list = utils.get_contours_and_images(final_img_bgr)
        except Exception as e:
            print(f"Error: {e}")
            return

        if not char_data_list:
            self.lbl_equation.config(text="Equation: No symbols detected")
            return

        # Prepare inputs for AI
        input_imgs = np.array([item[0] for item in char_data_list])
        predictions = self.model.predict(input_imgs)

        # --- 3. Map Labels & Apply Logic Fixes ---
        SYMBOL_MAP = {
            'add': '+', 'sub': '-', 'mul': '*', 'div': '/',
            'left': '(', 'right': ')',
            '0': '0', '1': '1', '2': '2', '3': '3', '4': '4',
            '5': '5', '6': '6', '7': '7', '8': '8', '9': '9'
        }

        raw_equation = ""

        for i, pred in enumerate(predictions):
            # A. Get AI Prediction
            idx = np.argmax(pred)
            label = config.LABELS[idx]
            confidence = np.max(pred)

            # B. Get Geometry Data
            w = char_data_list[i][1]
            h = char_data_list[i][2]
            holes = char_data_list[i][3]
            density = char_data_list[i][4]
            aspect_ratio = w / h

            #print(f"Char {i}: AI={label} ({confidence:.2f}) | Holes={holes} | Ratio={aspect_ratio:.2f}")

            # C. --- THE LOGIC POLICE (Fixing AI Errors) ---
            img_28 = char_data_list[i][5]
            col_sums = np.sum(img_28, axis=0)
            peak_col = np.argmax(col_sums)
            row_sums = np.sum(img_28, axis=1)
            peak_row = np.argmax(row_sums)

            # RULE 1: Fix '8' vs 'mul' (x)
            # '8' must have holes. 'x' must not.
            if label == '8' and holes == 0:
                print("   -> Fixed: AI said 8 but no holes. Changed to *")
                label = 'mul'
            elif label == 'mul' and holes >= 2:
                print("   -> Fixed: AI said mul but has holes. Changed to 8")
                label = '8'

            # RULE 2: Fix 'sub' (-) vs '1'
            # '-' is wide/flat. '1' is tall/thin.
            if label == '1' and aspect_ratio > 1.5:
                label = 'sub'
            elif label == 'sub' and aspect_ratio < 0.5:
                label = '1'

            is_col_centered = (11 <= peak_col <= 17)
            is_row_centered = (11 <= peak_row <= 17)

            # says 4, but is it +
            ys, xs = np.where(img_28 > 0)
            if label == '4' and len(xs) > 0:
                cx, cy = np.mean(xs), np.mean(ys)
                center_x = int(cx)
                center_y = int(cy)
                vertical_len = np.sum(img_28[:, center_x] > 0)
                horizontal_len = np.sum(img_28[center_y, :] > 0)

                if 11 <= cx <= 17 and 11 <= cy <= 17 and holes == 0:

                    row = img_28[14]  # middle row
                    horizontal_len = np.sum(row > 0)

                    if horizontal_len > 14 and vertical_len > 14:
                        label = 'add'

            # D. Build String
            symbol = SYMBOL_MAP.get(label, label)
            raw_equation += symbol

        print(f"Raw Equation: {raw_equation}")

        # --- 4. Clean & Calculate ---
        clean_eq = self.clean_equation(raw_equation)
        self.lbl_equation.config(text=f"Equation: {clean_eq}")

        try:
            # Safety check
            allowed = set("0123456789+-*/(). ")
            if not set(clean_eq).issubset(allowed):
                raise ValueError("Invalid chars")

            result = eval(clean_eq)

            # Format result
            if isinstance(result, float) and result.is_integer():
                result = int(result)
            elif isinstance(result, float):
                result = round(result, 4)

            self.lbl_result.config(text=f"Result: {result}", fg="green")

        except ZeroDivisionError:
            self.lbl_result.config(text="Result: Error (Div 0)", fg="red")
        except Exception:
            self.lbl_result.config(text="Result: Error", fg="red")
    def clean_equation(self, text):
        """
        Fixes common math syntax errors for Python's eval()
        """
        if not text: return ""

        # 1. Fix implicit multiplication: 2(3) -> 2*(3)
        # Digit followed by (
        text = re.sub(r'(\d)\(', r'\1*(', text)
        # ) followed by Digit
        text = re.sub(r'\)(\d)', r')*\1', text)
        # ) followed by (
        text = re.sub(r'\)\(', r')*(', text)

        # 2. Fix unclosed brackets
        left_count = text.count('(')
        right_count = text.count(')')
        if left_count > right_count:
            text += ')' * (left_count - right_count)

        # 3. Remove leading/trailing operators (e.g. +2+3 -> 2+3)
        text = text.strip("+-*/")

        return text


if __name__ == "__main__":
    root = tk.Tk()
    app = MathSolverApp(root)
    root.mainloop()
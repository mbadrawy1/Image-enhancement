import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from zipfile import ZipFile
from urllib.request import urlretrieve
from IPython.display import Image

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk

original_image_path = ""
background_image_path = ""

def download_and_unzip(url, save_path):
    print(f"Downloading and extracting assets....", end="")
    urlretrieve(url, save_path)
    try:
        with ZipFile(save_path) as z:
            z.extractall(os.path.split(save_path)[0])
        print("Done")
    except Exception as e:
        print("\nInvalid file.", e)

URL = r"https://www.dropbox.com/s/0oe92zziik5mwhf/opencv_bootcamp_assets_NB4.zip?dl=1"
asset_zip_path = os.path.join(os.getcwd(), "opencv_bootcamp_assets_NB4.zip")
if not os.path.exists(asset_zip_path):
    download_and_unzip(URL, asset_zip_path)



def adjust_brightness(image, value):
    matrix = np.ones(image.shape, dtype="uint8") * value
    return cv2.add(image, matrix)

def adjust_contrast(image, factor):
    return np.uint8(np.clip(cv2.multiply(np.float64(image), factor), 0, 255))

def apply_threshold(image, threshold_value, threshold_type):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    retval, thresholded = cv2.threshold(gray, threshold_value, 255, threshold_type)
    return thresholded

def apply_adaptive_threshold(image, max_value, adaptive_method, threshold_type, block_size, C):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return cv2.adaptiveThreshold(gray, max_value, adaptive_method, threshold_type, block_size, C)

def manipulate_logo(logo_image, background_image):
    logo_w = logo_image.shape[0]
    logo_h = logo_image.shape[1]
    aspect_ratio = logo_w / background_image.shape[1]
    dim = (logo_w, int(background_image.shape[0] * aspect_ratio))
    resized_background = cv2.resize(background_image, dim, interpolation=cv2.INTER_AREA)

    img_gray = cv2.cvtColor(logo_image, cv2.COLOR_RGB2GRAY)
    retval, img_mask = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
    img_mask_inv = cv2.bitwise_not(img_mask)

    img_background = cv2.bitwise_and(resized_background, resized_background, mask=img_mask)
    img_foreground = cv2.bitwise_and(logo_image, logo_image, mask=img_mask_inv)

    return cv2.add(img_background, img_foreground)


def open_original_image():
    global original_image_path
    filepath = filedialog.askopenfilename(
        initialdir="/",
        title="Select an Image",
        filetypes=(("Image files", "*.jpg *.jpeg *.png *.bmp"), ("all files", "*.*"))
    )
    if filepath:
        original_image_path = filepath
        show_image(filepath, original_image_label)

def open_background_image():
    global background_image_path
    filepath = filedialog.askopenfilename(
        initialdir="/",
        title="Select an Image",
        filetypes=(("Image files", "*.jpg *.jpeg *.png *.bmp"), ("all files", "*.*"))
    )
    if filepath:
        background_image_path = filepath
        show_image(filepath, background_image_label)

def show_image(filepath, label):
    img = Image.open(filepath)
    img.thumbnail((300, 300))
    photo = ImageTk.PhotoImage(img)
    label.config(image=photo)
    label.image = photo

def process_image():
    global original_image_path, background_image_path
    if not original_image_path:
        messagebox.showerror("Error", "Please select an original image.")
        return

    img = cv2.imread(original_image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    operation = operation_var.get()
    try:
        if operation == "brightness":
            value = int(brightness_entry.get())
            result = adjust_brightness(img_rgb, value)
        elif operation == "contrast":
            factor = float(contrast_entry.get())
            result = adjust_contrast(img_rgb, factor)
        elif operation == "threshold":
            threshold_value = int(threshold_entry.get())
            threshold_type = getattr(cv2, threshold_type_var.get())
            result = apply_threshold(img_rgb, threshold_value, threshold_type)
        elif operation == "adaptive_threshold":
            max_value = int(adaptive_max_value_entry.get())
            adaptive_method = getattr(cv2, adaptive_method_var.get())
            threshold_type = getattr(cv2, adaptive_threshold_type_var.get())
            block_size = int(adaptive_block_size_entry.get())
            C = int(adaptive_C_entry.get())
            result = apply_adaptive_threshold(img_rgb, max_value, adaptive_method, threshold_type, block_size, C)
        elif operation == "logo_manipulation":
            if not background_image_path:
                messagebox.showerror("Error", "Please select a background image for logo manipulation.")
                return
            background_img = cv2.imread(background_image_path)
            background_img_rgb = cv2.cvtColor(background_img, cv2.COLOR_BGR2RGB)
            result = manipulate_logo(img_rgb, background_img_rgb)
        else:
            messagebox.showerror("Error", "Invalid operation selected.")
            return

        plt.imshow(result)
        plt.title("Processed Image")
        plt.show()
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred during processing: {e}")

# --- GUI Setup ---

window = tk.Tk()
window.title("Image Enhancement")
window.geometry("800x600")  # Width x Height

# Image selection
original_image_label = tk.Label(window)
original_image_label.pack()
open_original_image_button = tk.Button(window, text="Open Original Image", command=open_original_image)
open_original_image_button.pack()

background_image_label = tk.Label(window)
background_image_label.pack()
open_background_image_button = tk.Button(window, text="Open Background Image (for logo manipulation)", command=open_background_image)
open_background_image_button.pack()

# Operation selection
operation_var = tk.StringVar(value="brightness")
operation_label = tk.Label(window, text="Select Operation:")
operation_label.pack()
operations = ["brightness", "contrast", "threshold", "adaptive_threshold", "logo_manipulation"]
operation_menu = ttk.Combobox(window, textvariable=operation_var, values=operations)
operation_menu.pack()

brightness_label = tk.Label(window, text="Brightness Value:")
brightness_label.pack()
brightness_entry = tk.Entry(window)
brightness_entry.pack()

contrast_label = tk.Label(window, text="Contrast Factor:")
contrast_label.pack()
contrast_entry = tk.Entry(window)
contrast_entry.pack()

threshold_label = tk.Label(window, text="Threshold Value:")
threshold_label.pack()
threshold_entry = tk.Entry(window)
threshold_entry.pack()
threshold_type_var = tk.StringVar(value="THRESH_BINARY")
threshold_types = ["THRESH_BINARY", "THRESH_BINARY_INV", "THRESH_TRUNC", "THRESH_TOZERO", "THRESH_TOZERO_INV"]
threshold_type_menu = ttk.Combobox(window, textvariable=threshold_type_var, values=threshold_types)
threshold_type_menu.pack()

adaptive_max_value_label = tk.Label(window, text="Adaptive Max Value:")
adaptive_max_value_label.pack()
adaptive_max_value_entry = tk.Entry(window)
adaptive_max_value_entry.pack()

adaptive_method_var = tk.StringVar(value="ADAPTIVE_THRESH_MEAN_C")
adaptive_methods = ["ADAPTIVE_THRESH_MEAN_C", "ADAPTIVE_THRESH_GAUSSIAN_C"]
adaptive_method_menu = ttk.Combobox(window, textvariable=adaptive_method_var, values=adaptive_methods)
adaptive_method_menu.pack()

adaptive_threshold_type_var = tk.StringVar(value="THRESH_BINARY")
adaptive_threshold_types = ["THRESH_BINARY", "THRESH_BINARY_INV"]
adaptive_threshold_type_menu = ttk.Combobox(window, textvariable=adaptive_threshold_type_var, values=adaptive_threshold_types)
adaptive_threshold_type_menu.pack()

adaptive_block_size_label = tk.Label(window, text="Block Size:")
adaptive_block_size_label.pack()
adaptive_block_size_entry = tk.Entry(window)
adaptive_block_size_entry.pack()

adaptive_C_label = tk.Label(window, text="C:")
adaptive_C_label.pack()
adaptive_C_entry = tk.Entry(window)
adaptive_C_entry.pack()

process_button = tk.Button(window, text="Process Image", command=process_image)
process_button.pack()

window.mainloop()
from flask import Flask, request, render_template, send_file, redirect, url_for, session
from PIL import Image, ImageOps, ImageFilter
import cv2
import numpy as np
import os

# Initialize Flask app
app = Flask(__name__, template_folder="templates", static_folder="static")
app.secret_key = 'your_secret_key'  # Needed for session handling

# Ensure the uploads folder exists
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Allowed image formats
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff', 'webp', 'avif'}

def allowed_file(filename):
    """Check if the uploaded file is a valid image format."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def welcome_page():
    return render_template('welcome.html')

@app.route('/login', methods=['GET', 'POST'])
def login_page():
    if request.method == 'POST':
        # Fake login for now (you can add proper authentication later)
        session['logged_in'] = True  
        return redirect(url_for('upload_page'))
    return render_template('login.html')

@app.route('/upload')
def upload_page():
      # Redirect to login if not logged in
    return render_template('upload.html')

@app.route('/convert', methods=['POST'])
def convert_image():
    if 'image' not in request.files:
        return "Error: No file uploaded"

    file = request.files['image']
    effect = request.form.get('effect', 'none')

    if file.filename == '':
        return "Error: No selected file"

    if not allowed_file(file.filename):
        return "Error: Unsupported file format"

    # Save the uploaded file
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    # Debugging: Check if file exists
    if not os.path.exists(filepath):
        return f"Error: File not found after saving - {filepath}"

    try:
        img = Image.open(filepath).convert("RGB")
        img.load()
    except Exception:
        return "Error: Invalid or corrupted image file"

    # Convert unsupported formats to PNG for processing
    if file.filename.lower().endswith(('avif', 'webp')):
        png_filepath = filepath.rsplit('.', 1)[0] + '.png'
        img.save(png_filepath)
        filepath = png_filepath

    # Apply selected effect
    img = apply_effect(img, filepath, effect)

    # Save Processed Image
    output_filename = "converted_" + file.filename.rsplit('.', 1)[0] + ".png"
    output_filepath = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
    img.save(output_filepath)

    return redirect(url_for('result_page', filename=output_filename))

@app.route('/result/<filename>')
def result_page(filename):
    return render_template('result.html', image_path=filename)

@app.route('/uploads/<filename>')
def processed_image(filename):
    return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename))

def apply_effect(image, filepath, effect):
    """Applies the selected effect to the image."""
    if effect == "grayscale":
        return image.convert("L")
    elif effect == "negative":
        return ImageOps.invert(image)
    elif effect == "sepia":
        return apply_sepia(image)
    elif effect == "sketch":
        return apply_sketch(filepath)
    elif effect == "cartoon":
        return apply_cartoon(filepath)
    elif effect == "oil_painting":
        return apply_oil_painting(filepath)
    elif effect == "hdr":
        return apply_hdr(filepath)
    elif effect == "emboss":
        return image.filter(ImageFilter.EMBOSS)
    elif effect == "edge_detection":
        return image.filter(ImageFilter.FIND_EDGES)
    elif effect == "detail_enhancement":
        return image.filter(ImageFilter.DETAIL)
    return image

# Function for sepia effect
def apply_sepia(image):
    """Apply sepia effect to an image."""
    sepia_filter = np.array([[0.393, 0.769, 0.189],
                             [0.349, 0.686, 0.168],
                             [0.272, 0.534, 0.131]])
    img_array = np.array(image)
    img_sepia = cv2.transform(img_array, sepia_filter)
    img_sepia = np.clip(img_sepia, 0, 255)
    return Image.fromarray(img_sepia.astype('uint8'))

# Function for sketch effect
def apply_sketch(image_path):
    """Convert an image into a pencil sketch."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    inverted = cv2.bitwise_not(img)
    blurred = cv2.GaussianBlur(inverted, (21, 21), 0)
    sketch = cv2.divide(img, 255 - blurred, scale=256)
    return Image.fromarray(sketch)

# Function for cartoon effect
def apply_cartoon(image_path):
    """Convert an image into a cartoon effect."""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 7)
    edges = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
    color = cv2.bilateralFilter(img, 9, 300, 300)
    cartoon = cv2.bitwise_and(color, color, mask=edges)
    return Image.fromarray(cv2.cvtColor(cartoon, cv2.COLOR_BGR2RGB))

# Function for oil painting effect
def apply_oil_painting(image_path):
    """Apply oil painting effect to an image."""
    img = cv2.imread(image_path)
    try:
        oil_paint = cv2.xphoto.oilPainting(img, 7, 1)
    except AttributeError:
        print("Error: OpenCV xphoto module not installed. Install opencv-contrib-python.")
        return Image.open(image_path)
    return Image.fromarray(cv2.cvtColor(oil_paint, cv2.COLOR_BGR2RGB))

# Function for HDR effect
def apply_hdr(image_path):
    """Apply HDR effect to an image."""
    img = cv2.imread(image_path)
    hdr = cv2.detailEnhance(img, sigma_s=12, sigma_r=0.15)
    return Image.fromarray(cv2.cvtColor(hdr, cv2.COLOR_BGR2RGB))

if __name__ == '__main__':
    app.run(debug=True)

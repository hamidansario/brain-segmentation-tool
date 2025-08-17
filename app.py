from flask import Flask, render_template, request
import os
import time
from utils.classical_methods import canny_edge_detection, otsu_thresholding
from utils.ai_methods import segment_brain, overlay_mask_on_image

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'dcm'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' in request.files and request.files['file'].filename != '':
            file = request.files['file']
            if file and allowed_file(file.filename):
                filename = file.filename
                save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(save_path)
                return render_template('index.html', filename=filename, method=None, message=f'File {filename} uploaded successfully!')
            else:
                return render_template('index.html', message='File type not allowed', method=None)
        
        elif 'method' in request.form and 'filename' in request.form:
            filename = request.form['filename']
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            if not filename or not os.path.isfile(image_path):
                return render_template(
                    "index.html",
                    message="Please upload an image first before running segmentation.",
                    filename="",  # Clear filename for safety
                    method=None,
                )    
            start_time = time.time()

            if request.form['method'] == 'unet':
                result_path = segment_brain(image_path)
                method = 'unet'
            elif request.form['method'] == 'canny':
                result_path = canny_edge_detection(image_path, 100, 200)
                method = 'canny'
            elif request.form['method'] == 'otsu':
                result_path = otsu_thresholding(image_path)
                method = 'otsu'
            else:
                return render_template('index.html', message='Invalid method', method=None)

            elapsed_time = time.time() - start_time
            result_filename = os.path.basename(result_path)
            overlay_filename = result_filename.replace('.png', '_overlayed.png')
            overlay_path = os.path.join(app.config['UPLOAD_FOLDER'], overlay_filename)
            overlay_mask_on_image(image_path, result_path, alpha=0.5, output_path=overlay_path)

            return render_template('index.html',
                                   filename=filename,
                                   result_image=result_filename,
                                   overlay_image=overlay_filename,
                                   message=f'{method.capitalize()} segmentation completed in {elapsed_time:.3f} seconds.',
                                   method=method,
                                   elapsed_time=elapsed_time)
    # GET
    return render_template('index.html', method=None)

if __name__ == '__main__':
    app.run(debug=True)

# app.py
import os
import numpy as np
import pandas as pd
import joblib
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max file size

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the pre-trained pipeline
try:
    pipeline = joblib.load('ml_pipeline.joblib')
except FileNotFoundError:
    pipeline = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    # Simpan data input untuk ditampilkan kembali
    input_data = {}
    prediction = None
    
    if request.method == 'POST':
        # Kumpulkan semua data input
        input_data = {
            'menteri': request.form.get('menteri', ''),
            'wamen': request.form.get('wamen', ''),
            'es_ia_kk': request.form.get('es_ia_kk', ''),
            # Tambahkan semua field numerik Anda di sini
            
            # Kolom kategorik
            'kode_eselon_i': request.form.get('kode_eselon_i', ''),
            'kode_korwil': request.form.get('kode_korwil', ''),
            'tipe_kantor': request.form.get('tipe_kantor', ''),
            'tipe_bangunan': request.form.get('tipe_bangunan', '')
        }
        
        # Cetak input_data untuk debugging
        print("Input Data:", input_data)
        
        # Konversi data untuk prediksi
        try:
            # Siapkan data untuk prediksi dengan konversi tipe
            predict_data = {}
            
            # Konversi field numerik
            numeric_fields = [
                'menteri', 'wamen', 'es_ia_kk', 'es_ia_nkk', 'es_ib', 
                'es_iia_kk', 'es_iia_nkk', 'es_iib', 'es_iii_kk', 
                'es_iii_nkk', 'es_iv_kk', 'es_iv_nkk', 'es_v', 
                'f-iv', 'f-iii', 'pelaksana', 'jumlah_pegawai', 
                'jumlah_pengunjung', 'luas_gk_eksisting', 'rkerja', 
                'rarsip', 'r_fungsional', 'toilet', 'r_server', 
                'r_layanan', 'lobby', 'nisbah'
            ]
            
            for field in numeric_fields:
                value = input_data.get(field, '')
                predict_data[field] = float(value) if value else 0
            
            # Tambahkan kolom kategorik
            categorical_fields = [
                'kode_eselon_i', 'kode_korwil', 
                'tipe_kantor', 'tipe_bangunan'
            ]
            
            for field in categorical_fields:
                predict_data[field] = input_data.get(field, '')
            
            # Konversi ke DataFrame
            input_df = pd.DataFrame([predict_data])
            
            # Cetak input_df untuk debugging
            print("Input DataFrame:")
            print(input_df)
            
            if pipeline:
                # Lakukan prediksi
                prediction = pipeline.predict(input_df)[0]
        
        except Exception as e:
            # Tangani kesalahan konversi
            print(f"Error during prediction: {e}")
            return render_template('index.html', 
                                   input_data=input_data, 
                                   error=f"Terjadi kesalahan: {str(e)}")
    
    # Render template dengan data input dan prediksi
    return render_template('index.html', 
                           input_data=input_data, 
                           prediction=prediction)

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('upload.html', error='No file part')
        
        file = request.files['file']
        
        if file.filename == '':
            return render_template('upload.html', error='No selected file')
        
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Read the uploaded Excel file
            try:
                df = pd.read_excel(filepath)
                # Process the dataframe similar to your original preprocessing
                return render_template('upload.html', data=df.head().to_html())
            except Exception as e:
                return render_template('upload.html', error=str(e))
    
    return render_template('upload.html')

@app.route('/save_model', methods=['POST'])
def save_model():
    try:
        # Save the pipeline using joblib
        joblib.dump(pipeline, 'ml_pipeline.joblib')
        return "Model saved successfully!"
    except Exception as e:
        return f"Error saving model: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)
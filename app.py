import os, cv2, torch, datetime, functools, time, base64, mysql.connector
import numpy as np
from flask import Flask, render_template, Response, request, redirect, url_for, session, flash, jsonify
from werkzeug.utils import secure_filename
from torch.serialization import add_safe_globals
from PIL import Image
import io

# --- INISIALISASI FLASK ---
app = Flask(__name__)
app.secret_key = "asrama_gh_key_secret"

UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# --- KONFIGURASI DATABASE ---
def get_db_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="12345", 
        database="asrama_gh"
    )

# --- LOAD MODEL YOLOv7 ---
try:
    torch.load = functools.partial(torch.load, weights_only=False)
    model = torch.hub.load('./yolov7', 'custom', 'best_model_asrama.pt', 
                           source='local', trust_repo=True)
    print("✅ BERHASIL: Model YOLOv7 siap digunakan!")
except Exception as e:
    print(f"❌ ERROR SAAT LOAD MODEL: {e}")

# --- KONFIGURASI DETEKSI ---
camera_active = False # Default mati saat startup
last_detection_status = {"unknown_detected": False, "timestamp": 0}

ASRAMA_LABELS = ['Arlan', 'Christiano', 'Fael', 'gafe', 'Jonathan', 'Kenneth', 'Kevin', 'Kia', 'Matthew', 'Natan', 'Reins', 'Sdyney']
LABEL_FIX = {'Arlan': 'Arlan Gorby Jonsend','Christiano': 'Christiano Oswald', 'Fael': 'Rafael Junio Kristanto', 'Reins': 'Sidney Alexander Junior Legi', 'Natan': 'Verench Russell Kaunang', 'Kia': 'Enzo Gustav Supit', 'Kevin': 'Hezekiah David Elijah Woy', 'Kenneth': 'Kevin Gunawan ', 'gafe': 'Jonathan', 'Sdyney': 'Gave Miracle Liando'}

# --- FUNGSI HELPER & DETEKSI ---
def log_to_db(nama, status_deteksi):
    """Mencatat aktivitas deteksi ke MySQL secara otomatis"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        # Mencegah spam log (1 entri per 30 detik untuk orang yang sama)
        cursor.execute("SELECT id FROM attendance_history WHERE nama = %s AND waktu > NOW() - INTERVAL 30 SECOND", (nama,))
        if not cursor.fetchone():
            cursor.execute("INSERT INTO attendance_history (nama, status) VALUES (%s, %s)", (nama, status_deteksi))
            conn.commit()
        cursor.close()
        conn.close()
    except Exception as e:
        print(f"⚠️ Gagal log DB: {e}")

def process_detections(frame, detections):
    detected_info = []
    for _, row in detections.iterrows():
        confidence = row['confidence']
        if confidence > 0.40:
            label_model = row['name']
            nama_tampilan = LABEL_FIX.get(label_model, label_model)
            if nama_tampilan in ASRAMA_LABELS and confidence > 0.50:
                color = (0, 255, 0) # Hijau untuk Mahasiswa Asrama
                log_to_db(nama_tampilan, "ASRAMA")
            else:
                nama_tampilan = "UNKNOWN"
                color = (0, 0, 255) # Merah untuk Orang Asing
                log_to_db("ORANG ASING", "STRANGER")

            x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 4)
            cv2.putText(frame, nama_tampilan, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
    return frame, detected_info

def gen_frames():
    cap = cv2.VideoCapture(0)
    # Gunakan resolusi paling rendah agar CPU tidak kerja keras
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    # Matikan buffer internal OpenCV agar tidak terjadi penumpukan frame lama
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    frame_count = 0
    while True:
        # 1. Bersihkan buffer: Ambil beberapa frame dan buang, ambil yang terakhir
        for _ in range(5):
            cap.grab()
        
        success, frame = cap.retrieve()
        if not success: break
        
        frame_count += 1
        
        # 2. Deteksi dilakukan hanya setiap 10 frame (Lebih jarang = Lebih lancar)
        if frame_count % 10 == 0:
            # Ubah ke RGB untuk model
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = model(img_rgb)
            frame, _ = process_detections(frame, results.pandas().xyxy[0])
            
        # 3. Kompresi kualitas JPEG diturunkan ke 30 (Sangat ringan untuk streaming)
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 30])
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

# --- ROUTES ---

@app.route('/')
def index():
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        u = request.form.get('username')
        p = request.form.get('password')
        
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        # Mencari user di tabel users
        cursor.execute("SELECT * FROM users WHERE username = %s AND password = %s", (u, p))
        user = cursor.fetchone()
        cursor.close()
        conn.close()
        
        if user:
            # PENTING: Simpan session agar tidak None saat dipanggil di profile
            session['user_id'] = user['id'] 
            session['user'] = user['username']
            session['role'] = user['role']
            session['nama_user'] = user['nama'] # Nama lengkap untuk filter riwayat
            
            if user['role'] == 'admin':
                return redirect(url_for('admin_dashboard'))
            elif user['role'] == 'operator':
                return redirect(url_for('operator_monitor'))
            elif user['role'] == 'mahasiswa':
                return redirect(url_for('mahasiswa_profile'))
        
        flash("Username atau Password salah!")
    return render_template('login.html')

@app.route('/admin/dashboard')
def admin_dashboard(): 
    if 'role' not in session or session.get('role') != 'admin':
        return redirect(url_for('login'))
    
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    # Mengambil data dari tabel mahasiswa
    cursor.execute("SELECT * FROM mahasiswa ORDER BY id DESC")
    mahasiswa_list = cursor.fetchall()
    cursor.close()
    conn.close()
    return render_template('admin_dashboard.html', mahasiswa=mahasiswa_list)

@app.route('/mahasiswa/profile')
def mahasiswa_profile():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)

    # 1. Ambil data profil user
    cursor.execute("SELECT * FROM users WHERE id = %s", (session['user_id'],))
    user_data = cursor.fetchone()

    if not user_data:
        conn.close()
        return "User tidak ditemukan. Silakan login kembali.", 404

    # 2. Ambil riwayat makan berdasarkan Nama Lengkap (untuk sinkronisasi kamera)
    cursor.execute("SELECT * FROM attendance_history WHERE nama = %s ORDER BY waktu DESC", (user_data['nama'],))
    history_list = cursor.fetchall()

    cursor.close()
    conn.close()

    return render_template('dashboard_mahasiswa.html', 
                           user=user_data, 
                           history=history_list, 
                           is_admin_view=False)

@app.route('/admin/view_profile/<int:mahasiswa_id>')
def admin_view_profile(mahasiswa_id):
    if 'role' not in session or session.get('role') != 'admin':
        return redirect(url_for('login'))
    
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    
    # Ambil data dari tabel mahasiswa
    cursor.execute("SELECT * FROM mahasiswa WHERE id = %s", (mahasiswa_id,))
    student = cursor.fetchone()
    
    if student:
        # Gunakan nama_lengkap dari tabel mahasiswa untuk cari riwayat
        cursor.execute("SELECT * FROM attendance_history WHERE nama = %s ORDER BY waktu DESC", (student['nama_lengkap'],))
        history = cursor.fetchall()
        
        # Mapping agar template dashboard_mahasiswa bisa mengenali (user.nama & user.username)
        student_mapped = {
            'nama': student['nama_lengkap'],
            'username': student['nomor_registrasi'],
            'foto_path': student.get('foto_path'),
            'status': student['status']
        }
    else:
        return "Mahasiswa tidak ditemukan", 404

    cursor.close()
    conn.close()
    return render_template('dashboard_mahasiswa.html', user=student_mapped, history=history, is_admin_view=True)

@app.route('/admin/mahasiswa/add', methods=['POST'])
def add_mahasiswa():
    nomor_reg = request.form.get('nomor_registrasi')
    nama = request.form.get('nama_lengkap')
    
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        # Simpan ke data induk mahasiswa
        cursor.execute("INSERT INTO mahasiswa (nomor_registrasi, nama_lengkap, status) VALUES (%s, %s, 'ASRAMA')", 
                       (nomor_reg, nama))
        # Buat akun login otomatis
        cursor.execute("INSERT INTO users (username, password, role, nama) VALUES (%s, %s, 'mahasiswa', %s)", 
                       (nomor_reg, 'asrama123', nama))
        conn.commit()
    except Exception as e:
        print(f"Error saat tambah: {e}")
        conn.rollback()
    finally:
        cursor.close()
        conn.close()
    return redirect(url_for('admin_dashboard'))

@app.route('/admin/mahasiswa/delete/<int:id>')
def delete_mahasiswa(id):
    if 'role' not in session or session.get('role') != 'admin':
        return redirect(url_for('login'))
        
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM mahasiswa WHERE id = %s", (id,))
    conn.commit()
    cursor.close()
    conn.close()
    return redirect(url_for('admin_dashboard'))

@app.route('/operator/monitor')
def operator_monitor():
    if 'role' not in session: return redirect(url_for('login'))
    
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM attendance_history ORDER BY waktu DESC LIMIT 20")
    history_list = cursor.fetchall()
    cursor.close()
    conn.close()
    return render_template('operator_monitor.html', history=history_list)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

# --- FEED & CAMERA ---
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/toggle_camera', methods=['POST'])
def toggle_camera():
    global camera_active
    camera_active = not camera_active
    return jsonify({'active': camera_active})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
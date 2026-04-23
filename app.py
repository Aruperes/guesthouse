import os, cv2, torch, functools, time, threading, mysql.connector
import numpy as np
from flask import Flask, render_template, Response, request, redirect, url_for, session, jsonify
from queue import Queue
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = "asrama_gh_key_2026"

# Konfigurasi Folder Upload Foto Mahasiswa
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# --- DATABASE CONNECTION ---
def get_db_connection():
    return mysql.connector.connect(
        host="localhost", user="root", password="12345", database="asrama_gh"
    )

# --- LOAD MODEL YOLOv7 ---
try:
    torch.load = functools.partial(torch.load, weights_only=False)
    model = torch.hub.load('./yolov7', 'custom', 'best_model_asrama.pt', source='local', trust_repo=True)
    model.conf = 0.4  
except Exception as e:
    print(f"Error Model: {e}")

LABEL_FIX = {
    'Reins': 'Sidney Alexander Junior Legi', 'Natan': 'Verench Russell Kaunang',
    'Kia': 'Enzo Gustav Supit', 'Kevin': 'Hezekiah David Elijah Woy',
    'Kenneth': 'Kevin Gunawan', 'gafe': 'Gave Miracle Liando',
    'Sdyney': 'Jonathan', 'Arlan': 'Arlan Gorby Jonsend',
    'Christiano': 'Christiano Oswald', 'Fael': 'Rafael Junio Kristanto'
}

# --- GLOBAL VARIABLES ---
camera_active = False
detected_info = {"nama": "-", "nim": "-", "status": "-", "is_stranger": False, "count_sesi": 0}
last_save_time = {}

# --- VIDEO STREAMING HELPER ---
class VideoStream:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.q = Queue()
        threading.Thread(target=self._reader, daemon=True).start()
    def _reader(self):
        while True:
            ret, frame = self.cap.read()
            if not ret: break
            if not self.q.empty():
                try: self.q.get_nowait()
                except: pass
            self.q.put(frame)
    def read(self): return self.q.get()

vs = VideoStream()

def save_attendance(nama):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("INSERT INTO attendance_history (nama, status) VALUES (%s, %s)", (nama, "ASRAMA"))
        conn.commit()
        cursor.close()
        conn.close()
    except: pass

stranger_start_time = None 

def gen_frames():
    global camera_active, detected_info, stranger_start_time
    while True:
        # Jika kamera OFF, kirim frame hitam
        if not camera_active:
            img = np.zeros((480, 640, 3), np.uint8)
            cv2.putText(img, "KAMERA NONAKTIF", (180, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            _, buffer = cv2.imencode('.jpg', img)
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(0.1)
            continue

        # Ambil frame dari kamera
        frame = vs.read()
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model(img_rgb)
        detections = results.pandas().xyxy[0]
        
        unknown_found_this_frame = False
        temp_data = {"nama": "-", "nim": "-", "status": "-", "is_stranger": False}
        
        # Proses deteksi
        if not detections.empty:
            for _, row in detections.iterrows():
                label = row['name']
                conf = row['confidence']
                x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
                
                if label in LABEL_FIX and conf > 0.85:
                    nama_asli = LABEL_FIX[label]
                    color = (0, 255, 0)
                    unknown_found_this_frame = False 
                    
                    if temp_data["nama"] == "-":
                        conn = get_db_connection()
                        cursor = conn.cursor(dictionary=True)
                        cursor.execute("SELECT nomor_registrasi FROM mahasiswa WHERE nama_lengkap = %s", (nama_asli,))
                        m_data = cursor.fetchone()
                        temp_data.update({
                            "nama": nama_asli,
                            "nim": m_data['nomor_registrasi'] if m_data else "-",
                            "status": "ASRAMA"
                        })
                        cursor.close()
                        conn.close()

                    if (nama_asli not in last_save_time) or (time.time() - last_save_time[nama_asli] > 60):
                        threading.Thread(target=save_attendance, args=(nama_asli,)).start()
                        last_save_time[nama_asli] = time.time()
                else:
                    nama_asli = "UNKNOWN"
                    color = (0, 0, 255)
                    unknown_found_this_frame = True

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 4)
                cv2.putText(frame, f"{nama_asli} {int(conf*100)}%", (x1, y1-15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        if unknown_found_this_frame:
            if stranger_start_time is None:
                stranger_start_time = time.time()
            
            duration = time.time() - stranger_start_time
            if duration > 2.0: 
                temp_data["is_stranger"] = True
            else:
                temp_data["is_stranger"] = False
        else:
            stranger_start_time = None 
            temp_data["is_stranger"] = False

        detected_info.update(temp_data)
        
        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

# --- ROUTES UMUM ---

@app.route('/')
def index():
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        u = request.form.get('username').strip()
        p = request.form.get('password').strip()
        r = request.form.get('role').strip()
        
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        cursor.execute("SELECT * FROM users WHERE username = %s AND password = %s AND role = %s", (u, p, r))
        user = cursor.fetchone()
        cursor.close()
        conn.close()
        
        if user:
            session['user_id'] = user['id']
            session['username'] = user['username']
            session['role'] = user['role']
            session['nama'] = user['nama']
            
            if user['role'] == 'admin':
                return redirect(url_for('admin_dashboard'))
            elif user['role'] == 'operator':
                return redirect(url_for('operator_monitor'))
            else:
                return redirect(url_for('mahasiswa_profile'))

        return render_template('login.html', error="NIM atau Password salah untuk role ini!")
            
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

# --- ROUTES ADMIN ---

@app.route('/admin/dashboard')
def admin_dashboard():
    if session.get('role') != 'admin': 
        return redirect(url_for('login'))
        
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    
    cursor.execute("SELECT * FROM mahasiswa")
    mhs = cursor.fetchall()

    cursor.execute("SELECT COUNT(*) as total FROM mahasiswa")
    total = cursor.fetchone()['total']
    
    cursor.execute("SELECT COUNT(*) as jml_asrama FROM mahasiswa WHERE status = 'ASRAMA'")
    asrama = cursor.fetchone()['jml_asrama']
    
    cursor.close()
    conn.close()

    return render_template('admin_dashboard.html', 
                           mahasiswa=mhs, 
                           total=total, 
                           asrama=asrama)

@app.route('/admin/add_mahasiswa', methods=['POST'])
def add_mahasiswa():
    nim = request.form.get('nim')
    nama = request.form.get('nama')
    status = request.form.get('status')
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("INSERT INTO mahasiswa (nomor_registrasi, nama_lengkap, status) VALUES (%s, %s, %s)", (nim, nama, status))
    cursor.execute("INSERT INTO users (username, password, nama, role) VALUES (%s, '123', %s, 'mahasiswa')", (nim, nama))
    conn.commit()
    return redirect(url_for('admin_dashboard'))

@app.route('/admin/view/<int:id>')
def admin_view_mhs(id):
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM mahasiswa WHERE id = %s", (id,))
    m = cursor.fetchone()
    cursor.execute("SELECT * FROM attendance_history WHERE nama = %s ORDER BY waktu DESC", (m['nama_lengkap'],))
    hist = cursor.fetchall()
    return render_template('dashboard_mahasiswa.html', user=m, history=hist, is_admin_view=True)

# --- ROUTES OPERATOR ---

@app.route('/operator/monitor')
def operator_monitor():
    if session.get('role') != 'operator': return redirect(url_for('login'))
    return render_template('operator_monitor.html')

@app.route('/get_status')
def get_status():
    global detected_info
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(DISTINCT nama) FROM attendance_history WHERE DATE(waktu) = CURDATE()")
    detected_info["count_sesi"] = cursor.fetchone()[0]
    cursor.close()
    conn.close()
    return jsonify(detected_info)

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/toggle_camera', methods=['POST'])
def toggle_camera():
    global camera_active
    camera_active = not camera_active
    return jsonify({'active': camera_active})

# --- ROUTES MAHASISWA ---

@app.route('/mahasiswa/profile')
def mahasiswa_profile():
    if session.get('role') != 'mahasiswa':
        return redirect(url_for('login'))

    nim_user = session.get('username')
    
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)

    cursor.execute("SELECT * FROM mahasiswa WHERE nomor_registrasi = %s", (nim_user,))
    m = cursor.fetchone()
    
    if not m:
        return "Data mahasiswa tidak ditemukan di database asrama. Hubungi Admin."

    cursor.execute("SELECT * FROM attendance_history WHERE nama = %s ORDER BY waktu DESC", (m['nama_lengkap'],))
    hist = cursor.fetchall()
    
    cursor.close()
    conn.close()
    
    return render_template('dashboard_mahasiswa.html', user=m, history=hist, is_admin_view=False)

@app.route('/mahasiswa/upload_face', methods=['POST'])
def upload_face():
    file = request.files.get('file')
    if file:
        filename = secure_filename(f"{session['username']}.jpg")
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("UPDATE mahasiswa SET foto_path = %s WHERE nomor_registrasi = %s", (filename, session['username']))
        conn.commit()
    return redirect(url_for('mahasiswa_profile'))

if __name__ == '__main__':
    app.run(debug=True, port=5000)
from flask import Flask, render_template, request, Response
from werkzeug.utils import secure_filename
import cv2
import os

application = Flask(__name__)
application.config['UPLOAD_FOLDER'] = os.path.realpath('.') + '/static/uploads'
application.config['MAX_CONTENT_PATH'] = 1000000

net = cv2.dnn.readNetFromDarknet('yolov4-obj.cfg', 'yolov4-obj_last.weights')

model = cv2.dnn_DetectionModel(net)
model.setInputParams(scale=1 / 255, size=(416, 416), swapRB=True)

cap = cv2.VideoCapture('video.mp4')

if not cap.isOpened():
    raise IOError("Cannot open video file")

# Daftar kelas yang digunakan dalam deteksi kendaraan
classes = ['truk', 'mobil', 'motor']

def gen_frames():
    while True:
        ret, img = cap.read()
        
        if not ret:
            break

        classIds, scores, boxes = model.detect(img, confThreshold=0.6, nmsThreshold=0.4)  
        
        print(classIds)
        
        truk = 0
        mobil = 0
        motor = 0
        
        for (classId, score, box) in zip(classIds, scores, boxes):
            cv2.rectangle(img, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]),
                        color=(0, 255, 0), thickness=2)
            
            if classId == 0:
                truk+= 1
            elif classId == 1:
                mobil+= 1
            elif classId == 2:
                motor+= 1
                
            text = classes[classId]
            cv2.putText(img, text, (box[0], box[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        color=(0, 255, 0), thickness=2)
            
        # Tampilkan jumlah kendaraan pada frame video
        cv2.putText(img, f"Jumlah truk: {truk}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(img, f"Jumlah mobil: {mobil}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(img, f"Jumlah motor: {motor}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        ret, buffer = cv2.imencode('.jpg', img)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()
    cv2.destroyAllWindows()

@application.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        f = request.files['file']

        global filename
        filename = application.config['UPLOAD_FOLDER'] + '/' + secure_filename(f.filename)

        try:
            f.save(filename)
            return render_template('form.html', filename=secure_filename(f.filename), notif='Upload Success')
        except:
            return render_template('upload_gagal.html')
    return render_template('form.html')

@application.route('/stream', methods=['GET', 'POST'])
def stream():
    if request.method == 'POST':
        return render_template('streaming.html')
    return render_template('upload_gagal.html')

@application.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@application.route("/about")
def about():
    return render_template("about.html")

if __name__ == '__main__':
    application.run(debug=True)

from flask import Flask, render_template, redirect, request, url_for
from werkzeug.utils import secure_filename
import os
import torch
import torchvision
from torch.utils.data import Dataset
from torchvision import transforms, models
from torch import nn
import numpy as np
import cv2
import face_recognition
import warnings
import shutil

warnings.filterwarnings("ignore")

UPLOAD_FOLDER = 'uploaded_files'
MODEL_PATH = 'model/df_model.pt'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__, template_folder='templates', static_folder='static')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ──────────────────────────────── Model Definition ────────────────────────────────

class Model(nn.Module):
    def __init__(self, num_classes, latent_dim=2048, lstm_layers=1, hidden_dim=2048, bidirectional=False):
        super(Model, self).__init__()
        model = models.resnext50_32x4d(pretrained=True)
        self.model = nn.Sequential(*list(model.children())[:-2])
        self.lstm = nn.LSTM(latent_dim, hidden_dim, lstm_layers, bidirectional)
        self.relu = nn.LeakyReLU()
        self.dp = nn.Dropout(0.4)
        self.linear1 = nn.Linear(2048, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        batch_size, seq_length, c, h, w = x.shape
        x = x.view(batch_size * seq_length, c, h, w)
        fmap = self.model(x)
        x = self.avgpool(fmap)
        x = x.view(batch_size, seq_length, 2048)
        x_lstm, _ = self.lstm(x, None)
        return fmap, self.dp(self.linear1(x_lstm[:, -1, :]))

sm = nn.Softmax()
im_size = 112
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

def predict(model, img):
    fmap, logits = model(img.to(torch.device('cpu')))
    logits = sm(logits)
    _, prediction = torch.max(logits, 1)
    confidence = logits[:, int(prediction.item())].item() * 100
    return [int(prediction.item()), confidence]

class validation_dataset(Dataset):
    def __init__(self, video_names, sequence_length=60, transform=None):
        self.video_names = video_names
        self.transform = transform
        self.count = sequence_length

    def __len__(self):
        return len(self.video_names)

    def __getitem__(self, idx):
        video_path = self.video_names[idx]
        frames = []
        for i, frame in enumerate(self.frame_extract(video_path)):
            faces = face_recognition.face_locations(frame)
            try:
                top, right, bottom, left = faces[0]
                frame = frame[top:bottom, left:right, :]
            except:
                pass
            frames.append(self.transform(frame))
            if len(frames) == self.count:
                break
        frames = torch.stack(frames)
        frames = frames[:self.count]
        return frames.unsqueeze(0)

    def frame_extract(self, path):
        vidObj = cv2.VideoCapture(path)
        success = 1
        while success:
            success, image = vidObj.read()
            if success:
                yield image

def detectFakeVideo(videoPath):
    transform_pipeline = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((im_size, im_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    dataset = validation_dataset([videoPath], sequence_length=20, transform=transform_pipeline)
    model = Model(2)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()
    prediction = predict(model, dataset[0])
    return prediction

# ──────────────────────────────── Routes ────────────────────────────────

@app.route('/', methods=['GET'])
def homepage():
    # Clear the preview video on page refresh (optional)
    preview_path = os.path.join('static', 'temp_preview.mp4')
    if os.path.exists(preview_path):
        os.remove(preview_path)
    return render_template("index.html")

@app.route('/Detect', methods=['POST'])
def DetectPage():
    if 'video' not in request.files:
        return redirect(url_for('homepage'))

    video = request.files['video']
    if video.filename == '':
        return redirect(url_for('homepage'))

    filename = secure_filename(video.filename)
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    video.save(video_path)

    # Copy to static folder for preview
    preview_path = os.path.join('static', 'temp_preview.mp4')
    shutil.copyfile(video_path, preview_path)

    # Predict
    prediction = detectFakeVideo(video_path)

    output = "REAL" if prediction[0] == 1 else "FAKE"
    confidence = prediction[1]

    result = {
        'output': output,
        'confidence': round(confidence, 2)
    }

    return render_template("index.html", result=result)

if __name__ == '__main__':
    app.run(debug=True, port=3000)

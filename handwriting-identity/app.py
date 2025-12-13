# app.py（修改版）
from flask import Flask, request, jsonify
from flask_cors import CORS  # ← 新增
import torch
import torch.nn as nn
from PIL import Image
import base64
import io
import numpy as np

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
    def forward(self, x):
        return self.net(x)

# 加载模型
device = torch.device("cpu")
model = MLP()
model.load_state_dict(torch.load("mnist_mlp.pth", map_location=device, weights_only=True))
model.eval()

MEAN = 0.1307
STD = 0.3081

app = Flask(__name__)
CORS(app)  # ← 允许所有来源跨域访问（开发用）

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json['image']
        if data.startswith('data:image'):
            data = data.split(',')[1]
        
        image_bytes = base64.b64decode(data)
        image = Image.open(io.BytesIO(image_bytes)).convert('L')
        image = image.resize((28, 28), Image.LANCZOS)
        img_array = np.array(image)
        img_array = 255 - img_array
        img_array = img_array.astype(np.float32) / 255.0
        img_array = (img_array - MEAN) / STD
        tensor = torch.tensor(img_array).view(1, -1).to(device)
        
        with torch.no_grad():
            output = model(tensor)
            pred = output.argmax(dim=1).item()
        
        return jsonify({"prediction": pred})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)
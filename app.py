from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
from PIL import Image
import io

# ===== Khởi tạo ứng dụng =====
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

# ===== Load mô hình =====
model = load_model("best_mn_finetune.keras")

# Nếu bạn có thứ tự nhãn từ training generator thì dùng lại
# class_names = ["Cat", "Dog"]
class_names = ["Cat", "Dog"]  # thay bằng list đúng thứ tự nếu cần

# ===== Hàm xử lý ảnh =====
def preprocess_image(img: Image.Image):
    img = img.resize((224, 224))
    img = image.img_to_array(img)
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)
    return img

# ===== Trang giao diện =====
@app.get("/", response_class=HTMLResponse)
async def root():
    with open("static/index.html", "r", encoding="utf-8") as f:
        html = f.read()
    return HTMLResponse(content=html)

# ===== API Dự đoán =====
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")
    img_array = preprocess_image(img)

    # ===== Dự đoán =====
    pred = model.predict(img_array)
    prob_pos = float(pred[0][0])
    prob_neg = 1.0 - prob_pos
    probs = [prob_neg, prob_pos]
    class_idx = int(prob_pos > 0.5)
    class_name = class_names[class_idx]

    return {
        "prediction": class_name,
        "probabilities": {
            class_names[0]: f"{probs[0]*100:.2f}%",
            class_names[1]: f"{probs[1]*100:.2f}%"
        }
    }

# Phân loại Chó vs Mèo với MobileNetV2

Dự án này là hệ thống phân loại hình ảnh để phân biệt chó và mèo sử dụng Transfer Learning với MobileNetV2.  
Mô hình được huấn luyện trên Dog and Cat Classification Dataset.

## Nội dung dự án

- Pipeline huấn luyện với TensorFlow/Keras
- Đánh giá bằng ma trận nhầm lẫn (confusion matrix) & báo cáo phân loại (classification report)
- Ứng dụng web FastAPI cho phép upload ảnh và dự đoán kết quả

## Yêu cầu

- Python 3.9+
- Khuyến nghị sử dụng GPU (đã test trên Colab T4)

## Cài đặt dependencies

```bash
pip install -r requirements.txt
```

## Cấu trúc dự án

```
vision-ai-intern-assignment/
│── app.py
│── train.ipynb
│── best_mn_finetune.keras
│── report.md
│── README.md
│── requirements.txt
│
├── static/
│   ├── style.css
│   └── index.html
```

## Huấn luyện mô hình

Mô hình được huấn luyện trên Google Colab (GPU T4).

### Các bước:

1. Load dataset từ Kaggle
2. Tiền xử lý: resize 224x224, chuẩn hóa, augmentation
3. Xây dựng mô hình MobileNetV2 (pretrained ImageNet)
4. Giai đoạn 1: Huấn luyện head (chỉ phần cuối)
5. Giai đoạn 2: Fine-tune các lớp cuối
6. Lưu mô hình tốt nhất thành `best_mn_finetune.keras`

**Độ chính xác trên tập validation đạt: ~98.9%**

## Đánh giá

- Validation Accuracy: ~98.9%
- Confusion Matrix & Classification Report: Precision/Recall cao cho cả hai lớp


## Chạy ứng dụng Web

Ứng dụng web được xây dựng bằng FastAPI.

**Chạy server tại máy local:**

```bash
cd vision-ai-intern-assignment
python -m uvicorn app:app --reload
```

**Mở trình duyệt:**  
http://127.0.0.1:8000

Bạn sẽ thấy trang web đơn giản, nơi có thể upload ảnh (chó/mèo) và nhận dự đoán.

## Ví dụ sử dụng

- Upload ảnh chó hoặc mèo.
- Ứng dụng sẽ tiền xử lý ảnh (resize 224x224, normalize, MobileNetV2 preprocess).
- Mô hình fine-tuned dự đoán xác suất.
- Kết quả hiển thị: nhãn dự đoán và độ tin cậy (confidence score).

## Báo cáo

Xem `report.md` để biết chi tiết:
- Pipeline
- Các thuật toán sử dụng
- Kết quả
- Vấn đề còn tồn tại & ý tưởng cải tiến

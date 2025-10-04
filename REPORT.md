# Báo cáo: Phân loại chó và mèo bằng MobileNetV2 (dùng gg colab để train với GPU-T4)
![alt text](image-4.png)
## 1. Pipeline
1. **Bộ dữ liệu**  
   - Kaggle dataset: [Bhavik Jikadara — Cat & Dog Classification Dataset](https://www.kaggle.com/datasets/bhavikjikadara/dog-and-cat-classification-dataset)   
   - Chia tập: 80% train, 20% validation  

2. **Tiền xử lý**  
   - Resize ảnh về 224x224  
   - Chuẩn hóa pixel về [0,1]  
   - Data augmentation: xoay, dịch, zoom, lật  

3. **Mô hình**  
   - Backbone: MobileNetV2 pretrained trên ImageNet  
   - Phần head:  
     - GlobalAveragePooling2D  
     - Dense(256, ReLU)  
     - Dropout(0.5)  
     - Dense(1, Sigmoid)  
   - Giai đoạn 1: Chỉ train phần head  
   - Giai đoạn 2: Fine-tune từ layer thứ 100 trở đi đến hết MobileNetV2 với learning rate thấp  
   ![alt text](image.png)

4. **Huấn luyện**  
   - Optimizer: Adam  
   - Loss: Binary Crossentropy  
   - Callbacks: EarlyStopping, ModelCheckpoint, ReduceLROnPlateau  
   - Epochs: 20 (giai đoạn 1) + 10 (giai đoạn 2)  

5. **Đánh giá**  
   - Accuracy trên tập validation: ~98.9%  
   - Confusion Matrix cho thấy số lỗi rất ít  
   - Classification Report: Precision/Recall/F1 ~0.99 
   ![alt text](image-1.png) 
   ![alt text](image-2.png)

6. **Demo**  
   - Hàm `predict_and_show()` dùng để dự đoán trên 1 ảnh mới (có thể demo trực tiếp theo như notebook hoặc chạy web demo)

---

## 2. Kết quả
- Accuracy giai đoạn 1: ~98.5%  
- Accuracy giai đoạn 2 (fine-tune): ~98.9%  
- Mô hình lưu lại: `best_mn_finetune.keras`  

---

## 3. **Web Demo**  
   - Xây dựng ứng dụng web bằng **FastAPI** để minh họa mô hình.  
   - Người dùng có thể upload ảnh (JPG/PNG), hệ thống sẽ:  
     1. Load mô hình đã huấn luyện (`best_mn_finetune.keras`)  
     2. Tiền xử lý ảnh (resize 224x224, normalize)  
     3. Chạy dự đoán và trả về kết quả **Chó** hoặc **Mèo** cùng xác suất.  
   - Giao diện đơn giản, chạy bằng lệnh:  
     ```bash
     uvicorn app:app --reload
     ```  
   - Có thể tích hợp thêm phần hiển thị ảnh và kết quả dự đoán trực quan trong trình duyệt. 
![alt text](image-3.png)
---

## 4. Hạn chế
- Mô hình luôn dự đoán chó/mèo, kể cả với ảnh ngoài domain (ví dụ: xe hơi, cây cối).  
- Dữ liệu có thể chứa ảnh nhiễu hoặc gán nhãn sai.  
- Mô hình phụ thuộc vào đặc trưng ImageNet, có thể thử backbone khác.  

---

## 5. Hướng cải tiến
- Thêm class "other" để xử lý ảnh ngoài domain.  
- Dùng backbone mạnh hơn: Vision Transformer.  
- Áp dụng regularization nâng cao: Label Smoothing.  
- Điều chỉnh siêu tham số để tối ưu tốt hơn.  

---



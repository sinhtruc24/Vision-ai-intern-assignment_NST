# BÁO CÁO  
## Exercise 2 – Text to Speech (TTS) Tiếng Việt  

---

## 1. Giới thiệu

### 1.1. Mục tiêu đề tài
Bài tập này nhằm xây dựng và mô tả **hệ thống Text-to-Speech (TTS) tiếng Việt** – một hệ thống có khả năng **chuyển đổi văn bản tiếng Việt thành giọng nói tự nhiên**, có thể tùy chỉnh theo **giọng, vùng miền, cảm xúc** và **tốc độ nói**.

Hệ thống được thiết kế hướng tới:
- **Chất lượng giọng nói cao**, tự nhiên và mượt mà.
- **Độ trễ thấp**, có thể triển khai trong thời gian thực.
- **Khả năng mở rộng**, thêm giọng mới hoặc phong cách nói mới dễ dàng.

---

## 2. Mô tả tổng quan hệ thống (TTS Pipeline)

Hệ thống TTS gồm 4 thành phần chính:

1. **Front-end (Tiền xử lý ngôn ngữ):**  
   Chuẩn hóa văn bản, phục hồi dấu, tách từ, chuyển văn bản thành chuỗi âm vị (phoneme).

2. **Acoustic Model:**  
   Dự đoán đặc trưng âm học (Mel-spectrogram) từ chuỗi âm vị và thông tin prosody (ngắt nghỉ, cao độ, năng lượng).

3. **Neural Vocoder:**  
   Biến Mel-spectrogram thành dạng sóng âm thanh (waveform).

4. **Post-processing và Deployment:**  
   Xử lý âm thanh cuối, chuẩn hóa volume, lưu file, đóng gói API.

Sơ đồ pipeline minh họa:

![TTS Pipeline](pipeline.png)

---

## 3. Các thành phần và thuật toán sử dụng

### 3.1. Front-end (Xử lý ngôn ngữ)

#### a. Chuẩn hóa văn bản (Text Normalization)
- Biến các dạng không đọc được (số, ngày, tiền tệ, ký hiệu) thành dạng đọc được.  
  Ví dụ: `123` → “một trăm hai mươi ba”.
- Kết hợp **luật, biểu thức chính quy (regex)** và **mô hình neural nhỏ** để xử lý các trường hợp đặc biệt.

#### b. Phục hồi dấu (Diacritics Restoration)
- Dùng mô hình **Transformer hoặc BiLSTM** để dự đoán dấu cho văn bản không dấu.
- Dữ liệu huấn luyện: cặp câu có dấu – không dấu được tạo tự động.

#### c. Tách từ và gán nhãn từ loại (Word Segmentation & POS Tagging)
- Dùng công cụ **PyVi hoặc mô hình CRF/Transformer** để tách từ chính xác.
- Thông tin POS hỗ trợ xác định cách phát âm và ngữ điệu phù hợp.

#### d. Chuyển chữ viết sang âm vị (G2P)
- Chuyển từng từ thành chuỗi **phoneme** và **thanh điệu (tone)**.
- Dựa trên quy tắc ngữ âm tiếng Việt, có cơ chế **lexicon lookup** và **seq2seq G2P model** để xử lý từ mượn.

#### e. Dự đoán Prosody sơ bộ
- Ước lượng ngắt nghỉ, nhấn trọng âm, cao độ (pitch), năng lượng (energy).
- Có thể thêm markup tương tự **SSML** để hỗ trợ điều khiển ngữ điệu.

---

### 3.2. Acoustic Model

- **Thuật toán sử dụng:** **FastSpeech 2**
  - Là mô hình **non-autoregressive**, huấn luyện song song, ổn định và nhanh hơn Tacotron2.
  - Dự đoán trực tiếp **duration**, **pitch**, và **energy** cho từng âm vị.
  - Input: chuỗi phoneme + tone + prosody embeddings.  
  - Output: **Mel-spectrogram** (80 chiều).

- **Loss function:**  
  - L1/MSE loss cho Mel-spectrogram.  
  - Duration, pitch và energy loss.  
  - Sử dụng optimizer **AdamW** và **scheduler Noam/annealing**.

---

### 3.3. Neural Vocoder

- **Thuật toán:** **HiFi-GAN**
  - Chuyển Mel-spectrogram thành sóng âm thanh (waveform).
  - Ưu điểm: tốc độ sinh nhanh, chất lượng cao, hỗ trợ real-time.
  - Có thể kết hợp **quantization (int8)** và **multi-band** để tối ưu cho thiết bị di động.

---

### 3.4. Post-processing & Deployment

- Chuẩn hóa âm lượng, lọc nhiễu nhẹ.  
- Xuất định dạng WAV/MP3, thêm metadata (giọng, vùng miền, style).  
- Triển khai qua **API REST/gRPC**.  
- Dùng **ONNX hoặc TFLite** cho edge deployment, giảm độ trễ.

---

## 4. Dữ liệu và huấn luyện

### 4.1. Dataset
- Gồm 10–20 giờ ghi âm tiếng Việt chuẩn, chất lượng cao (24 kHz, mono).  
- Ghi âm trong phòng cách âm, giọng đọc rõ ràng, có nhãn prosody và tone.  
- Nếu huấn luyện multi-speaker: cần 50+ giờ với nhiều giọng khác nhau.

### 4.2. Tiền xử lý dữ liệu
- Loại bỏ nhiễu, chuẩn hóa transcript.  
- Forced alignment giữa transcript và audio để trích duration/pitch/energy chính xác.

### 4.3. Huấn luyện
- Huấn luyện FastSpeech 2 để dự đoán Mel-spectrogram.  
- Huấn luyện HiFi-GAN vocoder từ Mel thật.  
- Sau đó kết hợp pipeline và tinh chỉnh (fine-tune) toàn hệ thống.

---

## 5. Vấn đề gặp phải và giải pháp

| Vấn đề | Nguyên nhân | Giải pháp |
|--------|--------------|-----------|
| Giọng bị “robot” | Dữ liệu ít hoặc không tự nhiên | Thu thêm dữ liệu chất lượng cao, augment nhẹ (speed/pitch) |
| Sai thanh điệu (tone) | Mất thông tin dấu hoặc pitch predictor yếu | Giữ tone xuyên pipeline, thêm f0 loss |
| Prosody khô cứng | Dự đoán đơn điệu | Thêm GST/VAE để học style/cảm xúc |
| Phát âm sai từ nước ngoài | Thiếu lexicon song ngữ | Thêm module transliteration + lookup |
| Latency cao | Mô hình autoregressive hoặc vocoder nặng | Dùng FastSpeech 2 + HiFi-GAN + quantization |

---

## 6. Đánh giá hệ thống

### 6.1. Đánh giá chủ quan (Human Evaluation)
- **MOS (Mean Opinion Score):** đánh giá độ tự nhiên, dễ nghe.  
- **CMOS:** so sánh với hệ thống baseline.

### 6.2. Đánh giá khách quan (Objective Metrics)
- **MCD (Mel Cepstral Distortion)** – so sánh phổ âm thanh.  
- **RMSE của F0**, **V/UV error** – đo độ chính xác ngữ điệu.  
- **CER/WER** – kiểm tra mức hiểu bằng ASR.

### 6.3. Bộ kiểm thử (Test Suite)
- Câu bao phủ toàn bộ âm vị và thanh điệu.  
- Trường hợp đặc biệt: số, ngày, tên riêng, từ tiếng Anh, câu không dấu.  
- Câu cảm thán, câu hỏi để đánh giá prosody.

---

## 7. Triển khai và tối ưu thời gian thực

- Dùng **FastSpeech 2 (non-autoregressive)** giúp sinh song song, giảm độ trễ.  
- **HiFi-GAN** chạy thời gian thực trên GPU/CPU nhẹ.  
- Tối ưu bằng **pruning**, **quantization**, và **ONNX Runtime**.  
- Hỗ trợ **streaming synthesis** cho chatbot/voice assistant.

---


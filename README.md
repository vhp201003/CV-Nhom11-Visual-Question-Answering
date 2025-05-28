# Visual Question Answering (VQA) - Hệ thống Hỏi Đáp Dựa trên Hình ảnh

Dự án này xây dựng một hệ thống Visual Question Answering (VQA) sử dụng deep learning để trả lời các câu hỏi bằng tiếng Việt dựa trên nội dung hình ảnh. Hệ thống kết hợp PhoBERT (mô hình ngôn ngữ tiếng Việt) và ResNet50 (mô hình thị giác máy tính) để hiểu và phân tích mối quan hệ giữa văn bản và hình ảnh.

## 📋 Mục Lục

- [Tổng quan](#tổng-quan)
- [Kiến trúc mô hình](#kiến-trúc-mô-hình)
- [Yêu cầu hệ thống](#yêu-cầu-hệ-thống)
- [Hướng dẫn tải dữ liệu](#hướng-dẫn-tải-dữ-liệu)
- [Xử lý dữ liệu](#xử-lý-dữ-liệu)
- [Hướng dẫn training](#hướng-dẫn-training)
- [Hướng dẫn inference](#hướng-dẫn-inference)
- [Cấu trúc thư mục](#cấu-trúc-thư-mục)
- [Kết quả](#kết-quả)

## 🎯 Tổng quan

Hệ thống VQA này có khả năng:
- Nhận đầu vào là một hình ảnh và câu hỏi bằng tiếng Việt
- Phân tích nội dung hình ảnh bằng computer vision
- Hiểu ngữ nghĩa câu hỏi tiếng Việt
- Trả về câu trả lời phù hợp dựa trên nội dung hình ảnh

**Nguồn dữ liệu gốc**: [Visual QA Dataset](https://visualqa.org/download.html)

## 🏗️ Kiến trúc mô hình

### Các thành phần chính:

1. **Image Encoder (ResNet50)**
   - Trích xuất đặc trưng từ hình ảnh
   - Output: vector đặc trưng 768 chiều

2. **Text Encoder (PhoBERT)**
   - Xử lý câu hỏi tiếng Việt
   - Model: `vinai/phobert-base`
   - Output: embedding vector cho câu hỏi

3. **Cross-Attention Mechanism**
   - Kết hợp thông tin từ hình ảnh và câu hỏi
   - Multi-head attention với 8 heads

4. **Classifier**
   - Dự đoán câu trả lời từ top 1000 câu trả lời phổ biến nhất

## 💻 Yêu cầu hệ thống

### Phần cứng:
- GPU với ít nhất 8GB VRAM (khuyến nghị RTX 3070 trở lên)
- RAM: 16GB trở lên
- Ổ cứng: 50GB dung lượng trống

### Phần mềm:
```bash
Python >= 3.8
PyTorch >= 1.9
CUDA >= 11.0
```

### Thư viện Python:
```bash
torch>=1.9.0
torchvision>=0.10.0
transformers>=4.20.0
googletrans==3.1.0a0
langdetect>=1.0.9
Pillow>=8.3.0
tqdm>=4.62.0
pandas>=1.3.0
matplotlib>=3.5.0
```

## 📥 Hướng dẫn tải dữ liệu

### Bước 1: Tải dữ liệu từ Visual QA

Truy cập [https://visualqa.org/download.html](https://visualqa.org/download.html) và tải về các file sau:

1. **Questions**:
   - `v2_OpenEnded_mscoco_val2014_questions.json`

2. **Annotations**:
   - `v2_mscoco_val2014_annotations.json`
   - `v2_mscoco_val2014_complementary_pairs.json`

3. **Images**:
   - `val2014.zip` (hình ảnh validation set)

### Bước 2: Cấu trúc thư mục dữ liệu

Sau khi tải về, tổ chức thư mục như sau:

```
data/
├── v2_OpenEnded_mscoco_val2014_questions.json
├── v2_mscoco_val2014_annotations.json
├── v2_mscoco_val2014_complementary_pairs.json
└── val2014/
    ├── COCO_val2014_000000000042.jpg
    ├── COCO_val2014_000000000073.jpg
    └── ...
```

### Bước 3: Kiểm tra dữ liệu

Đảm bảo các file có kích thước đúng:
- Questions: ~25MB
- Annotations: ~50MB
- Images: ~6.2GB (40,504 hình ảnh)

## 🔄 Xử lý dữ liệu

Quy trình xử lý dữ liệu được thực hiện theo thứ tự sau:

### 1. Phân tích và gộp dữ liệu (`processing_data.ipynb`)

```bash
jupyter notebook processing_data.ipynb
```

**Chức năng**:
- Phân tích cấu trúc các file JSON
- Gộp questions và annotations thành một file duy nhất
- Đổi tên file hình ảnh cho phù hợp
- Tạo file `merged_data.json`

### 2. Dịch sang tiếng Việt (`trans_script.py`)

```bash
python trans_script.py
```

**Chức năng**:
- Đọc dữ liệu từ `merged_data.json`
- Sử dụng Google Translate API để dịch questions và answers sang tiếng Việt
- Xử lý theo chunks (100 items/chunk) để tối ưu hiệu suất
- Có checkpoint để resume khi bị gián đoạn

**Cấu hình quan trọng**:
```python
CHUNK_SIZE = 100  # Kích thước mỗi chunk
INPUT_FILE = "merged_data.json"
OUTPUT_DIR = "translated_chunks"
```

### 3. Sửa lỗi và hoàn thiện dữ liệu (`fix_data_script.py`)

```bash
python fix_data_script.py
```

**Chức năng**:
- Kiểm tra và sửa các câu chưa được dịch đúng
- Sử dụng `langdetect` để phát hiện ngôn ngữ
- Dịch lại các phần chưa hoàn thiện
- Tạo file cuối cùng `output_merged_data_final.json`

### 4. Output cuối cùng

Sau khi hoàn thành, bạn sẽ có:
- `output_merged_data_final.json`: Dữ liệu questions/answers tiếng Việt
- `filtered_images/`: Thư mục chứa hình ảnh đã được lọc

## 🚀 Hướng dẫn Training

### 1. Chuẩn bị môi trường

```bash
# Cài đặt dependencies
pip install -r requirements.txt

# Kiểm tra GPU
python -c "import torch; print(torch.cuda.is_available())"
```

### 2. Cấu hình training (`visual-question-answering-vqa.ipynb`)

Mở notebook và điều chỉnh các thông số trong phần Config:

```python
# ===== Config =====
IMAGE_DIR = 'filtered_images'  # Đường dẫn thư mục ảnh
JSON_PATH = 'output_merged_data_final.json'  # File dữ liệu
PHOBERT_NAME = 'vinai/phobert-base'
BATCH_SIZE = 64         # Điều chỉnh theo GPU
NUM_EPOCHS = 5          # Số epoch training
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TRAIN_SPLIT = 0.8       # Tỷ lệ train/validation
TOP_K_ANSWERS = 1000    # Số lượng câu trả lời candidate
```

### 3. Khởi chạy training

```bash
# Chạy toàn bộ notebook
jupyter notebook visual-question-answering-vqa.ipynb

# Hoặc chạy script
python -c "exec(open('visual-question-answering-vqa.ipynb').read())"
```

### 4. Theo dõi quá trình training

Model sẽ in ra thông tin sau mỗi epoch:
```
[Epoch 1/5] Train Loss: 2.3456 | Val Loss: 2.1234 | Val Accuracy: 23.45%
[Epoch 2/5] Train Loss: 1.9876 | Val Loss: 1.8765 | Val Accuracy: 31.23%
...
```

### 5. Lưu model

Model được tự động lưu sau khi training:
- `vqa_model_final.pt`: Model weights
- `training_history.csv`: Lịch sử training

## 🔮 Hướng dẫn Inference

### 1. Load model đã train

```python
from transformers import AutoTokenizer, AutoModel
import torch
from PIL import Image
import torchvision.transforms as transforms

# Load model
model = VQAModel(num_answers=1000)
model.load_state_dict(torch.load('vqa_model_final.pt'))
model.eval()

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained('vinai/phobert-base')
```

### 2. Chuẩn bị dữ liệu đầu vào

```python
# Transform cho hình ảnh
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

# Load và preprocess hình ảnh
image = Image.open('path/to/image.jpg').convert('RGB')
image_tensor = transform(image).unsqueeze(0)

# Tokenize câu hỏi
question = "Có bao nhiêu người trong hình?"
q_tokens = tokenizer(question, padding='max_length', 
                    truncation=True, max_length=32, 
                    return_tensors='pt')
```

### 3. Thực hiện inference

```python
with torch.no_grad():
    logits = model(image_tensor, q_tokens)
    pred_idx = logits.argmax(dim=1).item()
    answer = idx_to_answer[pred_idx]
    print(f"Câu trả lời: {answer}")
```

### 4. Demo interactive

Sử dụng function có sẵn trong notebook:

```python
# Test với ảnh ngẫu nhiên
test_random_sample(model, dataset, tokenizer, transform, idx_to_answer, show_image=True)

# Test với ảnh tự chọn và đặt nhiều câu hỏi
test_random_image_with_questions(model, tokenizer, transform, idx_to_answer)
```

## 📁 Cấu trúc thư mục

```
CV-Nhom11/
├── README.md                                    # File hướng dẫn này
├── processing_data.ipynb                        # Notebook xử lý dữ liệu ban đầu
├── trans_script.py                             # Script dịch sang tiếng Việt
├── fix_data_script.py                          # Script sửa lỗi dữ liệu
├── visual-question-answering-vqa.ipynb         # Notebook training và inference
├── requirements.txt                            # Dependencies
├── data/                                       # Thư mục dữ liệu gốc
│   ├── v2_OpenEnded_mscoco_val2014_questions.json
│   ├── v2_mscoco_val2014_annotations.json
│   ├── v2_mscoco_val2014_complementary_pairs.json
│   └── val2014/
├── translated_chunks/                          # Chunks dữ liệu đã dịch
├── filtered_images/                            # Hình ảnh đã lọc
├── merged_data.json                           # Dữ liệu gộp ban đầu
├── output_merged_data_final.json              # Dữ liệu cuối cùng (tiếng Việt)
├── vqa_model_final.pt                         # Model đã train
└── training_history.csv                       # Lịch sử training
```

## 📊 Kết quả

### Hiệu suất mô hình:
- **Dataset**: ~40,000 cặp hình ảnh-câu hỏi-đáp án
- **Top-1000 answers accuracy**: ~35-45% (tùy thuộc vào số epoch)
- **Training time**: ~2-4 giờ trên GPU RTX 3070
- **Inference time**: ~100ms/sample

### Một số ví dụ kết quả:

| Hình ảnh | Câu hỏi | Dự đoán | Ground Truth |
|----------|---------|---------|--------------|
| 🏠 | Có bao nhiêu cửa sổ? | 2 | 2 |
| 🐕 | Con vật này là gì? | chó | chó |
| 🚗 | Màu xe là gì? | đỏ | đỏ |

### Hạn chế:
- Độ chính xác còn hạn chế do dữ liệu dịch máy
- Một số câu trả lời bị nhiễu do lỗi dịch
- Model chỉ có thể trả lời trong top-1000 answers

## 🔧 Troubleshooting

### Lỗi thường gặp:

1. **CUDA out of memory**:
   ```bash
   # Giảm batch size
   BATCH_SIZE = 32  # hoặc 16
   ```

2. **Google Translate API limit**:
   ```bash
   # Thêm delay giữa các request
   time.sleep(1)
   ```

3. **File not found**:
   ```bash
   # Kiểm tra đường dẫn
   ls -la data/
   ```

## 🤝 Đóng góp

Dự án này được phát triển bởi nhóm 11 - môn Computer Vision. Chào mừng các đóng góp để cải thiện hiệu suất và tính năng của hệ thống.

## 📝 License

Dự án này sử dụng cho mục đích học tập và nghiên cứu.

---

**Chúc bạn thành công với dự án VQA! 🎉**

*Nếu gặp vấn đề, hãy tạo issue hoặc liên hệ với nhóm phát triển.*

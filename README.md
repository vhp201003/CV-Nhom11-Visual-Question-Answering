# Visual Question Answering (VQA) - Hệ thống Hỏi Đáp Dựa trên Hình ảnh

Dự án này phát triển một hệ thống Visual Question Answering (VQA) ứng dụng công nghệ deep learning để trả lời các câu hỏi bằng tiếng Việt dựa trên nội dung hình ảnh đầu vào. Hệ thống tích hợp PhoBERT (mô hình xử lý ngôn ngữ tự nhiên tiếng Việt) và ResNet50 (mô hình thị giác máy tính) nhằm hiểu và phân tích mối quan hệ phức tạp giữa thông tin văn bản và thị giác.

## Mục Lục

- [Tổng quan](#tổng-quan)
- [Kiến trúc mô hình](#kiến-trúc-mô-hình)
- [Yêu cầu hệ thống](#yêu-cầu-hệ-thống)
- [Hướng dẫn tải dữ liệu](#hướng-dẫn-tải-dữ-liệu)
- [Xử lý dữ liệu](#xử-lý-dữ-liệu)
- [Hướng dẫn training](#hướng-dẫn-training)
- [Hướng dẫn inference](#hướng-dẫn-inference)
- [Cấu trúc thư mục](#cấu-trúc-thư-mục)
- [Kết quả và đánh giá](#kết-quả-và-đánh-giá)
- [Xử lý sự cố và khắc phục lỗi](#xử-lý-sự-cố-và-khắc-phục-lỗi)
- [Hạn chế và hướng phát triển](#hạn-chế-và-hướng-phát-triển)
- [Đóng góp và phát triển](#đóng-góp-và-phát-triển)
- [Giấy phép sử dụng](#giấy-phép-sử-dụng)

## Tổng quan

## Tổng quan

Hệ thống VQA được phát triển nhằm giải quyết bài toán hiểu và trả lời câu hỏi dựa trên nội dung hình ảnh, với các khả năng chính:

- Xử lý đầu vào đa phương thức: hình ảnh và câu hỏi bằng tiếng Việt
- Phân tích và trích xuất đặc trưng từ nội dung hình ảnh sử dụng kỹ thuật computer vision
- Hiểu ngữ nghĩa và ngữ cảnh của câu hỏi tiếng Việt
- Tổng hợp thông tin đa phương thức để đưa ra câu trả lời chính xác

**Nguồn dữ liệu**: [Visual QA Dataset](https://visualqa.org/download.html)

## Kiến trúc mô hình

## Kiến trúc mô hình

Hệ thống VQA được thiết kế theo kiến trúc đa phương thức (multimodal) với các thành phần chính:

### 1. Bộ mã hóa hình ảnh (Image Encoder)
- **Backbone**: ResNet50 với pre-trained weights
- **Chức năng**: Trích xuất đặc trưng không gian từ hình ảnh đầu vào
- **Output**: Tensor đặc trưng với số chiều 768

### 2. Bộ mã hóa văn bản (Text Encoder)
- **Model**: PhoBERT (`vinai/phobert-base`)
- **Chức năng**: Xử lý và mã hóa câu hỏi tiếng Việt thành vector đặc trưng
- **Output**: Embedding vectors cho biểu diễn ngữ nghĩa

### 3. Cơ chế Cross-Attention
- **Kiến trúc**: Multi-head attention với 8 attention heads
- **Chức năng**: Tích hợp và kết hợp thông tin từ cả hai modality (hình ảnh và văn bản)
- **Mục đích**: Tạo ra biểu diễn đa phương thức thống nhất

### 4. Bộ phân loại (Classifier)
- **Kiến trúc**: Fully connected layer
- **Chức năng**: Dự đoán câu trả lời từ tập 1000 câu trả lời phổ biến nhất
- **Output**: Phân phối xác suất trên không gian câu trả lời

## Yêu cầu hệ thống

## Yêu cầu hệ thống

### Yêu cầu phần cứng
- **GPU**: Tối thiểu 8GB VRAM (khuyến nghị NVIDIA RTX 3070 hoặc cao hơn)
- **Bộ nhớ RAM**: 16GB trở lên
- **Dung lượng lưu trữ**: 50GB không gian trống

### Yêu cầu phần mềm
```bash
Python >= 3.8
PyTorch >= 1.9
CUDA >= 11.0
```

### Thư viện phụ thuộc
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

## Hướng dẫn tải dữ liệu

## Hướng dẫn tải dữ liệu

### Bước 1: Thu thập dữ liệu từ Visual QA Dataset

Truy cập trang web chính thức [https://visualqa.org/download.html](https://visualqa.org/download.html) và tải xuống các file dữ liệu sau:

**1. File câu hỏi (Questions)**:
- `v2_OpenEnded_mscoco_val2014_questions.json`

**2. File chú thích (Annotations)**:
- `v2_mscoco_val2014_annotations.json`
- `v2_mscoco_val2014_complementary_pairs.json`

**3. Dữ liệu hình ảnh (Images)**:
- `val2014.zip` (tập validation của MS COCO dataset)

### Bước 2: Tổ chức cấu trúc dữ liệu

Sau khi tải xuống hoàn tất, tổ chức dữ liệu theo cấu trúc thư mục sau:

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

### Bước 3: Xác thực tính toàn vẹn dữ liệu

Kiểm tra kích thước file để đảm bảo dữ liệu được tải xuống hoàn chỉnh:
- File Questions: khoảng 25MB
- File Annotations: khoảng 50MB
- Thư mục Images: khoảng 6.2GB (chứa 40,504 hình ảnh)

## Xử lý dữ liệu

## Xử lý dữ liệu

Quy trình xử lý dữ liệu được thực hiện theo pipeline gồm ba giai đoạn chính:

### Giai đoạn 1: Phân tích và gộp dữ liệu (`processing_data.ipynb`)

```bash
jupyter notebook processing_data.ipynb
```

**Mục tiêu và chức năng**:
- Phân tích cấu trúc và schema của các file JSON đầu vào
- Thực hiện liên kết dữ liệu giữa questions và annotations
- Chuẩn hóa tên file hình ảnh cho tương thích với hệ thống
- Xuất ra file `merged_data.json` chứa dữ liệu tích hợp

### Giai đoạn 2: Dịch máy sang tiếng Việt (`trans_script.py`)

```bash
python trans_script.py
```

**Mục tiêu và chức năng**:
- Đọc và xử lý dữ liệu từ file `merged_data.json`
- Áp dụng Google Translate API để dịch questions và answers sang tiếng Việt
- Xử lý dữ liệu theo batches (100 mẫu/batch) để tối ưu hiệu suất và tránh rate limiting
- Triển khai cơ chế checkpoint để phục hồi quá trình khi bị gián đoạn

**Tham số cấu hình quan trọng**:
```python
CHUNK_SIZE = 100                    # Kích thước batch xử lý
INPUT_FILE = "merged_data.json"     # File dữ liệu đầu vào
OUTPUT_DIR = "translated_chunks"    # Thư mục lưu kết quả tạm thời
```

### Giai đoạn 3: Kiểm tra chất lượng và hoàn thiện (`fix_data_script.py`)

```bash
python fix_data_script.py
```

**Mục tiêu và chức năng**:
- Kiểm tra và đánh giá chất lượng dịch thuật
- Sử dụng thư viện `langdetect` để phát hiện ngôn ngữ và xác định các trường hợp dịch không thành công
- Thực hiện dịch lại các phần tử chưa hoàn thiện
- Tạo file cuối cùng `output_merged_data_final.json` với dữ liệu đã được làm sạch

### Kết quả đầu ra

Sau khi hoàn thành pipeline xử lý dữ liệu:
- `output_merged_data_final.json`: Dataset chính chứa questions/answers bằng tiếng Việt
- `filtered_images/`: Thư mục chứa hình ảnh đã được lọc và chuẩn hóa

## Hướng dẫn Training

## Hướng dẫn Training

### Bước 1: Chuẩn bị môi trường phát triển

```bash
# Cài đặt các thư viện phụ thuộc
pip install -r requirements.txt

# Kiểm tra tính khả dụng của GPU
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

### Bước 2: Cấu hình hyperparameters (`visual-question-answering-vqa.ipynb`)

Mở notebook và điều chỉnh các tham số training trong phần cấu hình:

```python
# ===== Cấu hình Training =====
IMAGE_DIR = 'filtered_images'                  # Đường dẫn thư mục hình ảnh
JSON_PATH = 'output_merged_data_final.json'    # File dữ liệu đã xử lý
PHOBERT_NAME = 'vinai/phobert-base'            # Pretrained PhoBERT model
BATCH_SIZE = 64                                # Kích thước batch (điều chỉnh theo GPU)
NUM_EPOCHS = 5                                 # Số epoch training
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TRAIN_SPLIT = 0.8                              # Tỷ lệ phân chia train/validation
TOP_K_ANSWERS = 1000                           # Số lượng câu trả lời ứng viên
```

### Bước 3: Thực thi quá trình training

```bash
# Chạy notebook training hoàn chỉnh
jupyter notebook visual-question-answering-vqa.ipynb

# Hoặc chạy trực tiếp từ command line
python -c "exec(open('visual-question-answering-vqa.ipynb').read())"
```

### Bước 4: Giám sát quá trình training

Hệ thống sẽ hiển thị thông tin training sau mỗi epoch:
```
[Epoch 1/5] Train Loss: 2.3456 | Val Loss: 2.1234 | Val Accuracy: 23.45%
[Epoch 2/5] Train Loss: 1.9876 | Val Loss: 1.8765 | Val Accuracy: 31.23%
[Epoch 3/5] Train Loss: 1.7654 | Val Loss: 1.6543 | Val Accuracy: 38.67%
...
```

### Bước 5: Lưu trữ model và metadata

Sau khi hoàn thành training, hệ thống tự động lưu:
- `vqa_model_final.pt`: Trọng số của model đã được training
- `training_history.csv`: Lịch sử training bao gồm loss và accuracy theo từng epoch

## Hướng dẫn Inference

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

## Cấu trúc thư mục

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

## Kết quả và đánh giá

### Hiệu suất hệ thống:
- **Dataset**: ~40,000 cặp hình ảnh-câu hỏi-đáp án
- **Top-1000 answers accuracy**: ~35-45% (tùy thuộc vào số epoch)
- **Training time**: ~2-4 giờ trên GPU RTX 3070
- **Inference time**: ~100ms/sample

### Mẫu kết quả thực nghiệm:

| Loại hình ảnh | Câu hỏi | Dự đoán | Kết quả thực tế |
|---------------|---------|---------|-----------------|
| Kiến trúc | Có bao nhiêu cửa sổ? | 2 | 2 |
| Động vật | Con vật này là gì? | chó | chó |
| Phương tiện | Màu xe là gì? | đỏ | đỏ |

### Hạn chế và thách thức:
- Độ chính xác còn hạn chế do chất lượng dữ liệu từ quá trình dịch máy
- Một số câu trả lời không chính xác do lỗi trong quá trình dịch thuật
- Phạm vi trả lời bị giới hạn trong tập 1000 câu trả lời phổ biến nhất

## Xử lý sự cố và khắc phục lỗi

### Các lỗi thường gặp và giải pháp:

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

## Hạn chế và hướng phát triển

### Hạn chế hiện tại

1. **Chất lượng dịch thuật**: Việc sử dụng Google Translate để chuyển đổi dữ liệu từ tiếng Anh sang tiếng Việt có thể dẫn đến mất mát thông tin ngữ nghĩa và ngữ cảnh.

2. **Phạm vi câu trả lời**: Hệ thống chỉ có thể đưa ra câu trả lời trong tập 1000 câu trả lời phổ biến nhất, hạn chế khả năng xử lý các câu hỏi phức tạp hoặc cần câu trả lời chi tiết.

3. **Hiệu suất mô hình**: Độ chính xác vẫn còn hạn chế so với các hệ thống VQA tiên tiến khác, đặc biệt với các câu hỏi yêu cầu suy luận phức tạp.

### Hướng phát triển tương lai

1. **Cải thiện chất lượng dữ liệu**: Thu thập và xây dựng dataset VQA tiếng Việt gốc, giảm thiểu sự phụ thuộc vào dịch máy.

2. **Mở rộng tập câu trả lời**: Tăng số lượng câu trả lời ứng viên và phát triển cơ chế sinh câu trả lời tự do.

3. **Tích hợp mô hình tiên tiến**: Áp dụng các kiến trúc mới như Vision Transformer (ViT) và các mô hình ngôn ngữ lớn (LLM) cho tiếng Việt.

4. **Tối ưu hiệu suất**: Cải thiện tốc độ inference và giảm yêu cầu tài nguyên tính toán.

## Đóng góp và phát triển

Dự án này được phát triển bởi nhóm 11 - môn Computer Vision. Chúng tôi hoan nghênh các đóng góp từ cộng đồng nhằm cải thiện hiệu suất và mở rộng tính năng của hệ thống.

## Giấy phép sử dụng

Dự án này được phát triển phục vụ mục đích học tập và nghiên cứu học thuật.

---

## Kết luận

Hệ thống Visual Question Answering này đại diện cho một bước tiến quan trọng trong việc ứng dụng công nghệ AI đa phương thức cho ngôn ngữ tiếng Việt. Mặc dù vẫn còn những hạn chế nhất định, dự án đã chứng minh khả năng tích hợp thành công các mô hình deep learning tiên tiến để giải quyết bài toán phức tạp về hiểu ngôn ngữ tự nhiên và thị giác máy tính.

*Để được hỗ trợ kỹ thuật hoặc báo cáo lỗi, vui lòng tạo issue trong repository hoặc liên hệ trực tiếp với nhóm phát triển.*

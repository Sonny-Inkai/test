# ViT5 Vietnamese Legal Joint Entity and Relation Extraction

Hệ thống trích xuất thực thể và quan hệ đồng thời (Joint Entity and Relation Extraction) cho văn bản luật Việt Nam sử dụng mô hình ViT5.

## 🚀 Tính năng chính

- **Joint Extraction**: Trích xuất thực thể và quan hệ trong một lần xử lý
- **Domain-specific**: Tối ưu cho văn bản luật Việt Nam
- **ViT5-based**: Sử dụng ViT5 (Vietnamese T5) thay vì BART như REBEL gốc
- **Special Tokens**: Hỗ trợ các loại thực thể và quan hệ đặc trung cho luật pháp
- **End-to-end**: Không cần pipeline phức tạp

## 📋 Cấu trúc Special Tokens

### Entity Types:
- `<ORGANIZATION>`: Tổ chức, cơ quan (Bộ Tài chính, Ủy ban nhân dân...)
- `<LOCATION>`: Địa điểm (Việt Nam, Hà Nội...)
- `<DATE/TIME>`: Thời gian (01/01/2021, năm 2025...)
- `<LEGAL_PROVISION>`: Điều khoản pháp luật (Điều 5, Luật Doanh nghiệp 2020...)

### Relation Types:
- `<Relates_To>`: Quan hệ liên quan chung
- `<Effective_From>`: Có hiệu lực từ
- `<Applicable_In>`: Áp dụng tại
- `<Amended_By>`: Được sửa đổi bởi

## 📊 Format dữ liệu

### Input Format:
```json
{
    "legal_1": {
        "title": "Điều 2 01/2014/NQLT/CP-UBTƯMTTQVN",
        "input_text": "Điều 2 01/2014/NQLT/CP-UBTƯMTTQVN hướng dẫn phối hợp thực hiện một số quy định của pháp luật về hòa giải ở cơ sở...",
        "extracted_relations": "<LEGAL_PROVISION> Điều 2 01/2014/NQLT/CP-UBTƯMTTQVN <ORGANIZATION> Mặt trận Tổ quốc Việt Nam <Relates_To>"
    }
}
```

### Output Format:
```
<LEGAL_PROVISION> Điều 5 Nghị định 15/2020/NĐ-CP <ORGANIZATION> Ủy ban nhân dân <Relates_To> <LEGAL_PROVISION> Luật Doanh nghiệp 2020 <DATE/TIME> 01/01/2021 <Effective_From>
```

## 🛠️ Cài đặt

### 1. Cài đặt dependencies:
```bash
pip install -r requirements_vit5.txt
```

### 2. Chuẩn bị dữ liệu:
- Đặt file `dataset.json` trong thư mục `/kaggle/input/vietnamese-legal-finetune-dataset/`
- Hoặc sửa đường dẫn trong `train_vit5_legal.py`

## 🏃 Chạy Training

### 1. Training cơ bản:
```bash
python train_vit5_legal.py
```

### 2. Training với custom config:
Sửa `config` trong file `train_vit5_legal.py`:
```python
config = {
    'data_path': "/path/to/your/data",
    'model_name': "VietAI/vit5-base",  # hoặc "VietAI/vit5-large"
    'batch_size': 4,
    'learning_rate': 3e-4,
    'max_epochs': 10,
    'max_steps': 10000,
    'gradient_clip_val': 1.0,
    'accumulate_grad_batches': 4,
    'precision': 16  # Mixed precision để tiết kiệm GPU
}
```

## 🧪 Testing Model

### 1. Test với checkpoint:
```bash
python test_vit5_legal.py --model_path checkpoints/vit5-legal-epoch=05-val_loss=0.1234.ckpt
```

### 2. Test với data file:
```bash
python test_vit5_legal.py --model_path checkpoints/best_model/ --test_data test_data.json
```

### 3. Test với custom parameters:
```bash
python test_vit5_legal.py \
    --model_path checkpoints/best_model/ \
    --max_length 512 \
    --num_beams 5 \
    --test_data my_test_data.json
```

## 📈 Monitoring Training

### Training sẽ hiển thị:
- **Loss theo epoch**: Train loss và validation loss
- **Sample generation**: Sau mỗi epoch sẽ generate sample để kiểm tra
- **Model checkpoints**: Tự động save best models trong `checkpoints/`
- **Early stopping**: Dừng sớm nếu validation loss không cải thiện

### Ví dụ output:
```
================================================================================
EPOCH 2 - SAMPLE GENERATION:
Input: extract relations: Điều 5 Nghị định 15/2020/NĐ-CP quy định về tổ chức và hoạt động của Ủy ban nhân dân xã, phường, thị trấn. Ủy ban nhân dân có trách nhiệm thực hiện các nhiệm vụ được giao.
Generated: <LEGAL_PROVISION> Điều 5 Nghị định 15/2020/NĐ-CP <ORGANIZATION> Ủy ban nhân dân <Relates_To>
================================================================================
```

## ⚙️ Configuration

### GPU Settings:
- **T4x2 Kaggle**: Tự động detect và sử dụng 2 GPUs
- **Single GPU**: devices=1
- **CPU**: accelerator='cpu'

### Memory Optimization:
- **Mixed precision**: precision=16
- **Gradient accumulation**: accumulate_grad_batches=4
- **Batch size**: Điều chỉnh theo GPU memory

### Model Variants:
```python
# Base model (nhanh hơn)
model_name = "VietAI/vit5-base"

# Large model (chính xác hơn)
model_name = "VietAI/vit5-large"
```

## 🔧 Troubleshooting

### 1. Lỗi GPU:
```python
# Fix cho PyTorch Lightning mới
trainer = pl.Trainer(
    accelerator='gpu',  # thay vì gpus=2
    devices=2           # thay vì gpus=2
)
```

### 2. Lỗi AdamW:
```python
# Sử dụng torch.optim thay vì transformers
from torch.optim import AdamW
# thay vì from transformers import AdamW
```

### 3. Memory issues:
```python
# Giảm batch size
batch_size = 2

# Tăng gradient accumulation
accumulate_grad_batches = 8

# Giảm sequence length
max_source_length = 256
max_target_length = 128
```

### 4. Slow training:
```python
# Giảm num_workers nếu CPU bottleneck
num_workers = 0

# Bật pin_memory
pin_memory = True

# Sử dụng mixed precision
precision = 16
```

## 📊 Performance Tips

### 1. **Hyperparameter tuning**:
```python
learning_rate = 3e-4      # Bắt đầu với giá trị này
warmup_steps = 1000       # 10% của total steps
weight_decay = 0.01       # Regularization
```

### 2. **Data preprocessing**:
- Normalize text trước khi train
- Check quality của extracted_relations
- Balance data nếu cần thiết

### 3. **Evaluation**:
- Monitor cả train và validation loss
- Check sample generation quality
- Evaluate trên test set riêng

## 📁 File Structure

```
.
├── train_vit5_legal.py       # Script training chính
├── test_vit5_legal.py        # Script testing
├── requirements_vit5.txt     # Dependencies
├── dataset.json              # Sample data
├── README_ViT5_Legal.md      # Documentation này
├── checkpoints/              # Model checkpoints (tự tạo)
│   ├── vit5-legal-epoch-xx-val_loss-x.xxxx.ckpt
│   └── last.ckpt
├── train_data.json          # Training split (tự tạo)
├── val_data.json            # Validation split (tự tạo)
└── test_data.json           # Test split (tự tạo)
```

## 🎯 Ví dụ sử dụng

### Training:
```python
# Chỉ cần chạy
python train_vit5_legal.py
```

### Testing:
```python
# Load model và test
from test_vit5_legal import ViT5LegalTester

tester = ViT5LegalTester("checkpoints/best_model/")
result = tester.extract_relations("Điều 5 Luật Doanh nghiệp 2020 có hiệu lực từ 01/01/2021")
print(result)
```

## 🤝 Đóng góp

1. Fork repository
2. Tạo feature branch
3. Commit changes
4. Push và tạo Pull Request

## 📄 License

MIT License - Xem file LICENSE để biết thêm chi tiết.

## 🙏 Acknowledgments

- **REBEL**: Inspiration từ paper "REBEL: Relation Extraction By End-to-end Language generation"
- **ViT5**: Vietnamese T5 model từ VietAI
- **PyTorch Lightning**: Framework training
- **Transformers**: Hugging Face library 
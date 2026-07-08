# Brain Tumor MRI Classification

Project huấn luyện và đánh giá mô hình phân loại ảnh MRI u não với 30 lớp. Notebook chính sử dụng backbone DINOv2 ViT-B/14 và lưu weight tốt nhất để phục vụ inference hoặc đánh giá lại.

## Dataset

Dataset sử dụng trong project:

- Kaggle: [Brain Tumor MRI Images - 30 Classes](https://www.kaggle.com/datasets/fernando2rad/brain-tumor-mri-images-30-classes)

Có thể tải trực tiếp bằng `kagglehub` trong notebook:

```python
from pathlib import Path
import kagglehub

DATASET_PATH = Path(kagglehub.dataset_download('fernando2rad/brain-tumor-mri-images-30-classes'))
print(f'Dataset: {DATASET_PATH}')
```

Dataset gồm ảnh MRI được chia thành 30 class. Trong notebook, dữ liệu được chia theo seed cố định `RANDOM_SEED = 32` với tỉ lệ train/validation/test.

## Model

Weight DINO tốt nhất được lưu trên Hugging Face:

- Hugging Face: [rubypham292/Brain-Tumor-Classification-DINO](https://huggingface.co/rubypham292/Brain-Tumor-Classification-DINO)
- File cần tải: `dino_best_weights.pt`

Sau khi tải, đặt file weight vào thư mục:

```text
model/dino_best_weights.pt
```

## Cấu trúc project

```text
app_img_classification/
├── dataset/                 # Thư mục chứa dataset nếu tải thủ công
├── model/                   # Thư mục chứa weight model
│   └── dino_best_weights.pt # Weight tải từ Hugging Face
├── notebook/
│   ├── dinov2-vit-b14z-final.ipynb
│   └── conv-next-b-failed-ver.ipynb
├── LICENSE
└── README.md
```

## Cài đặt môi trường

Các thư viện chính được dùng trong notebook:

```bash
pip install torch torchvision timm pandas numpy scikit-learn matplotlib seaborn pillow kagglehub
```

Nếu chạy trên Kaggle, nên bật GPU trước khi chạy notebook.

## Notebook chính

Notebook DINOv2:

```text
notebook/dinov2-vit-b14z-final.ipynb
```

Notebook này thực hiện các bước chính:

1. Tải dataset bằng `kagglehub`.
2. Chia train/validation/test theo 30 class.
3. Khởi tạo model `vit_base_patch14_dinov2` bằng `timm`.
4. Train theo 2 phase:
   - Phase 1: freeze backbone, train classification head.
   - Phase 2: unfreeze 4 block cuối của backbone.
5. Lưu best weight, classification report, confusion matrix và model summary.

## Kết quả đầu ra

Sau khi chạy notebook, các file kết quả được lưu trong thư mục output của notebook, ví dụ:

```text
dino_v2_m_outputs/
├── best_weights.pt
├── training_log.csv
├── classification_report.txt
├── confusion_matrix.png
├── learning_curves.png
└── model_summary.json
```

## License

Project sử dụng giấy phép MIT. Xem chi tiết trong file [LICENSE](LICENSE).

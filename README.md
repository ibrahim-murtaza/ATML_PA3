## Reproducibility for Knowledge Distillation

### Pre-trained Model Weights

All trained model weights are available on Google Drive for reproducibility:

#### **Part 1: Logit Matching Variants**
**[Google Drive - Part 1 Models](https://drive.google.com/drive/u/0/folders/1tgeyVd4cINIv-yxH3bdkdpJmwYm-SLb8)**

Models included:
- `teacher_vgg16_cifar100.pth` - Teacher VGG-16 (74.00% top-1)
- `student_vgg11_SI_best.pth` - Independent Student (66.12%)
- `student_vgg11_LM_best.pth` - Logit Matching (69.27%)
- `student_vgg11_LS_best.pth` - Label Smoothing (66.99%)
- `student_vgg11_DKD_best.pth` - Decoupled KD (69.33%)

#### **Part 2: Feature-Based Methods**
**[Google Drive - Part 2 Models](https://drive.google.com/drive/u/0/folders/11vcfbksIYz777EcJ7Ctq-oGu-9E1LIKK)**

Models included:
- `student_vgg11_HINTS_best.pth` - FitNets/HINTS (67.93%)
- `student_vgg11_CRD_best.pth` - Contrastive Distillation (69.77%)

#### **Part 5: Color Invariance Transfer**
**[Google Drive - Part 5 Models](https://drive.google.com/drive/u/0/folders/13HEJC29Dd_VH4tRLZJmkT2ULLgnAi6ue)**

Models included:
- `teacher_vgg16_color_invariant.pth` - Color-invariant Teacher (72.80%)
- `student_vgg11_CRD_color_best.pth` - Student from color-invariant teacher (70.16%)

#### **Part 6: Teacher Size Impact**
**[Google Drive - Part 6 Models](https://drive.google.com/drive/u/0/folders/1eSrquSi0qVo1s_b73v9vIVkGwLCg9kYg)**

Models included:
- `teacher_vgg19_cifar100_best.pth` - Teacher VGG-19 (73.83%)
- `student_vgg11_LM_VGG19_best.pth` - Student from VGG-19 teacher (68.26%)

### Results Summary

#### Overall Method Comparison

| Model | Method | Top-1 Acc (%) | Top-5 Acc (%) | Training Time (min) |
|-------|--------|---------------|---------------|---------------------|
| **Teacher (VGG-16)** | - | 74.00 | 90.54 | - |
| Student (VGG-11) | SI | 66.12 | 87.11 | 18.77 |
| Student (VGG-11) | LM | 69.27 | 88.07 | 20.71 |
| Student (VGG-11) | LS | 66.99 | 87.73 | 19.60 |
| Student (VGG-11) | DKD | 69.33 | 86.67 | 20.57 |
| Student (VGG-11) | HINTS | 67.93 | 87.16 | 22.14 |
| Student (VGG-11) | **CRD** | **69.77** | 87.98 | 46.98 |

**Best Method:** CRD achieves highest top-1 accuracy (69.77%) but requires 2.3× training time compared to DKD (69.33%, 20.57 min).

#### Teacher Size Comparison

| Teacher → Student | Teacher Acc (%) | Student Acc (%) | Gap (%) |
|-------------------|-----------------|-----------------|---------|
| VGG-16 → VGG-11 | 74.00 | 69.27 | 4.73 |
| VGG-19 → VGG-11 | 73.83 | 68.26 | 5.57 |

**Finding:** Larger teacher (VGG-19) produces worse student due to capacity mismatch.

#### Color Invariance Transfer

| Model | Acc Original (%) | Acc Jittered (%) | Agreement (%) | Acc Drop (%) |
|-------|------------------|------------------|---------------|--------------|
| SI (Baseline) | 66.12 | 46.23 | 53.31 | 19.89 |
| S_CRD (Control) | 69.77 | 50.52 | 58.71 | 19.25 |
| **S_CRD_color** | **70.16** | **55.86** | **65.47** | **14.30** |

**Finding:** CRD successfully transfers color invariance from teacher to student without explicit color augmentation during student training.

### Usage

To use the pre-trained weights:

1. Download models from the appropriate Google Drive link
2. Place them in: `Knowledge_Distillation/results/task3_X/models/` (where X is the part number)
3. Load in PyTorch:
```python
import torch
from torchvision.models import vgg11_bn

model = vgg11_bn(weights=None)
model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, 100)
checkpoint = torch.load('path/to/student_vgg11_CRD_best.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
```
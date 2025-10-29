## Reproducibility

### Pre-trained Model Weights

The trained model weights for Task 3.1 (Logit Matching Variants) can be downloaded from:

**[Google Drive - Task 3.1 Models](https://drive.google.com/drive/u/0/folders/1tgeyVd4cINIv-yxH3bdkdpJmwYm-SLb8)**

To use these weights:

1. Download all model files from the Google Drive folder
2. Place them in: `Knowledge_Distillation/results/task3_1/models/`
3. The models include:
   - `student_SI_best.pth` - Independent Student (VGG-11)
   - `student_LM_best.pth` - Logit Matching (KD)
   - `student_LS_best.pth` - Label Smoothing
   - `student_DKD_best.pth` - Decoupled Knowledge Distillation

### Results Summary

| Model | Method | Top-1 Acc | Top-5 Acc |
|-------|--------|-----------|-----------|
| Teacher (VGG-16) | - | 74.00% | 90.54% |
| Student (VGG-11) | SI | 66.12% | 87.11% |
| Student (VGG-11) | LM | 69.27% | 88.07% |
| Student (VGG-11) | LS | 66.99% | 87.73% |
| Student (VGG-11) | DKD | 69.57% | 86.71% |
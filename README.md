Salient Object Detection (SOD)

This project implements a full Salient Object Detection (SOD) pipeline from scratch. The goal was to build an end-to-end system that loads and preprocesses a dataset, trains a CNN-based encoder–decoder model, evaluates the results using standard segmentation metrics, and provides a small demo for inference and visualization.

Dataset

The dataset used is DUTS, which is one of the standard datasets for Salient Object Detection. The preprocessing steps include:
resizing to 224×224,
normalization to [0, 1],
custom dataset split: train (70%), validation (15%), test (15%),
data augmentations:
horizontal flip,
brightness/contrast adjustment.

Model Architecture

The model is implemented completely from scratch.
It is a U-Net-style encoder–decoder with skip connections.Main components:
Encoder:
4 blocks of DoubleConv (Conv → BN → ReLU ×2),
MaxPooling after each block,
Bottleneck with DoubleConv.
Decoder:
4 upsampling stages  ConvTranspose2d,
skip connections from encoder layers,
DoubleConv after each concatenation.
Final layer:
1×1 convolution,
Sigmoid activation for binary mask output
Input shape: 3 × 224 × 224,
Output shape: 1 × 224 × 224

Training

Optimizer: Adam
Loss function:
BCE + 0.5 × (1 − IoU),
Validation at the end of each epoch,
Best model saving based on validation loss,
Learning-rate scheduler ,
20-25 Epochs,
Bonus feature included.

Results

The best model:
Precision: 0.8471
Recall: 0.8232
F1-score: 0.8110
IoU: 0.7227

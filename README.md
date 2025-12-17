# SGD vs Adam on VGG16 (CIFAR-100)

This repository contains a simple experimental comparison of **SGD** and **Adam** optimizers when training a **VGG16** convolutional neural network from scratch on the **CIFAR-100** dataset.

The focus is on **early training behavior** under comparable conditions.

---

## Experiment Summary

- **Dataset:** CIFAR-100 (100 classes)
- **Model:** VGG16 (trained from scratch)
- **Batch size:** 64
- **Epochs:** 5
- **Loss:** Cross-Entropy

The only difference between the two runs is the **optimizer configuration**.

---

## Optimizers

### SGD
```python
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.005,
    momentum=0.9
)
```
### Adam
```python
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.0001
)
```
Different learning rates were used to ensure stable training for each optimizer.
## Results

- Both optimizers trained stably.

- SGD achieved slightly higher accuracy in early epochs.

- Adam nearly matched SGD by epoch 4.

- After 5 epochs, SGD showed a small accuracy advantage.

- This is an early-stage comparison; results may differ with longer training or learning-rate scheduling.

## Notes

- No pretrained weights were used.

- No weight decay was applied.

- This experiment is intended for educational and exploratory purposes.


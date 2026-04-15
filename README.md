# prompted-drywall-segmentation

## 📌 Overview

This project implements a **text-conditioned segmentation model** for drywall quality inspection.

Given:

* an input image
* a natural language prompt

the model outputs a **binary segmentation mask** for:

* 🧩 *Taping Area (Drywall Joints)*
* 🪨 *Cracks*

---

## 🎯 Goal

Train a model that understands prompts like:

* `"segment crack"`
* `"segment wall crack"`
* `"segment taping area"`
* `"segment joint tape"`

and produces accurate segmentation masks.

---

## 📂 Datasets

### 1️⃣ Drywall Taping Area

🔗 https://universe.roboflow.com/objectdetect-pu6rn/drywall-join-detect

### 2️⃣ Crack Dataset

🔗 https://universe.roboflow.com/fyp-ny1jt/cracks-3ii36

---

## 🧠 Model

* **Model Used:** CLIPSeg (`CIDAS/clipseg-rd64-refined`)
* **Type:** Prompt-based segmentation
* **Why:** Supports natural language conditioning

---

## ⚙️ Training Details

| Parameter  | Value      |
| ---------- | ---------- |
| Epochs     | 20         |
| Batch Size | 4          |
| Image Size | 352×352    |
| Optimizer  | AdamW      |
| LR         | 3e-4       |
| Loss       | BCE + Dice |
| Device     | GPU (T4)   |
| Seed       | 42         |

---

## 📊 Results

### 🔹 Cracks

| Prompt             | mIoU   | Dice   |
| ------------------ | ------ | ------ |
| segment crack      | 0.5559 | 0.6955 |
| segment wall crack | 0.5557 | 0.6954 |

### 🔹 Taping Area

| Prompt              | mIoU   | Dice   |
| ------------------- | ------ | ------ |
| segment taping area | 0.6631 | 0.7825 |
| segment joint tape  | 0.6636 | 0.7828 |

---

## 🌍 Overall Performance

* **Global mIoU:** 0.5730
* **Global Dice:** 0.7094

---

## 🖼️ Visual Results

### Example Predictions

| Input                             | Ground Truth                      | Prediction                        |
| --------------------------------- | --------------------------------- | --------------------------------- |
| ![](outputs/visuals/example1.png) | ![](outputs/visuals/example2.png) | ![](outputs/visuals/example3.png) |

*(Add your 3–4 best outputs here)*

---

## ⚡ Performance

| Metric          | Value        |
| --------------- | ------------ |
| Training Time   | ~96 minutes  |
| Inference Speed | ~96 ms/image |
| Model Size      | ~603 MB      |

---

## ❌ Failure Cases

* Thin cracks may be partially missed
* Texture noise causes false positives
* Slight boundary inaccuracies

---

## 📁 Output Format

* PNG masks
* Single-channel
* Values: `{0, 255}`

Example:

```
123__segment_crack.png
```

---

## 🚀 How to Run

```bash
git clone https://github.com/yourusername/prompted-drywall-segmentation
cd prompted-drywall-segmentation
pip install -r requirements.txt
```

### Train

```bash
python train.py
```

### Inference

```bash
python inference.py --image path.jpg --prompt "segment crack"
```

---

## 📌 Key Highlights

✔ Prompt-based segmentation
✔ Works across multiple defect types
✔ Robust to prompt variations
✔ Real-time capable (~96 ms/image)

---

## 👤 Author

* Your Name

---

## ⭐ If you like this project, give it a star!

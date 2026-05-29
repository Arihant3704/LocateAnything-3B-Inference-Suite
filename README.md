---
title: LocateAnything
emoji: 💬
colorFrom: yellow
colorTo: purple
sdk: gradio
sdk_version: 6.5.1
python_version: "3.10.13"
app_file: app.py
pinned: false
hf_oauth: true
hf_oauth_scopes:
- inference-api
tags:
- arxiv:2605.27365
---

# LocateAnything-3B Inference Suite

Official Gradio inference web application for **LocateAnything-3B**, a state-of-the-art vision-language model designed for object detection and visual grounding with natural language.

📄 **Paper:** [arxiv.org/abs/2605.27365](https://arxiv.org/abs/2605.27365)  
🤗 **Model Hub:** [nvidia/LocateAnything-3B](https://huggingface.co/nvidia/LocateAnything-3B)
---

## 📐 Model Architecture

```mermaid
graph TD
    classDef default fill:#F9F9FB,stroke:#E2E8F0,stroke-width:1px,color:#1E293B
    classDef input fill:#F0FDFA,stroke:#14B8A6,stroke-width:1.5px,color:#0F766E
    classDef component fill:#EFF6FF,stroke:#3B82F6,stroke-width:1.5px,color:#1D4ED8
    classDef process fill:#FFF7ED,stroke:#F97316,stroke-width:1.5px,color:#C2410C
    classDef output fill:#F0FDF4,stroke:#22C55E,stroke-width:1.5px,color:#15803D

    subgraph Inputs ["Input Stream"]
        I1["🖼️ Input Image"]
        I2["💬 Text Query / Target Categories"]
    end

    subgraph Visual ["Vision Encoding (Moon-ViT)"]
        V1["Visual Feature Extraction"]
        V2["Patch Projection & Alignment"]
    end

    subgraph Model ["LocateAnything-3B Core"]
        M1["Qwen2 3B Backbone"]
        M2["4-Bit NF4 Quantized Language Model"]
    end

    subgraph Decoding ["Decoding Engine"]
        D1["MTP Parallel Box Decoding<br/>(Predicts 4 box coords in parallel)"]
        D2["AR Autoregressive Fallback<br/>(Text / refinement tokens)"]
    end

    subgraph Outputs ["Outputs"]
        O1["📄 Structured Text Response"]
        O2["🎯 Bounding Boxes drawn on Image"]
    end

    I1 --> V1
    V1 --> V2
    V2 --> M1
    I2 --> M1
    M1 --> M2
    M2 --> D1
    M2 --> D2
    D1 --> O2
    D2 --> O1

    class I1,I2 input;
    class V1,V2,M1,M2 component;
    class D1,D2 process;
    class O1,O2 output;
```

---

## 🖥️ Running UI Interface

Below is the interface running inference on target query categories, showcasing the visual bounding box coordinates and the execution stats trace:

![LocateAnything UI Screenshot](assets/ui_screenshot.png)

---

## 🌟 Features

* **Multi-Format Task Support**: Locate objects, perform visual grounding, GUI layout parsing, OCR box detection, and pointing tasks.
* **MTP (Multi-Token Prediction) Parallel Decoding**: Speeds up inference by decoding multiple bounding box coordinates in parallel.
* **AR (Autoregressive) Fallback**: Seamless transition to standard autoregressive decoding when refinement is needed.
* **4-Bit NF4 Quantization**: Integrated `bitsandbytes` memory optimization, reducing required GPU memory from **10.5 GB** to only **~3.9 GB** total process VRAM.
* **Rich Decoding Visualization Trace**: Real-time stats panel displaying:
  * Inference execution time (in seconds)
  * Real-time generation throughput (FPS / Tokens per second)
  * Step details, predicted boxes, and AR fallback counts

---

## 🚀 Setup & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/localizeanything.git
cd localizeanything
```

### 2. Install Dependencies
Create a virtual environment and install the required Python packages:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

> [!NOTE]
> For 4-bit quantized loading, ensure you have a CUDA-compatible GPU, and that `bitsandbytes` and `accelerate` are successfully installed.

---

## 💻 Running the App

### Standard Inference (GPU Mode)
To run the Gradio application using the model weights (it will download automatically from Hugging Face on first launch):
```bash
python3 app.py
```

### Mock Mode (For Local/CPU Testing)
To run the application instantly without downloading the 3B model weights:
```bash
MOCK_MODE=1 python3 app.py
```

Open `http://127.0.0.1:7860` in your web browser.

---

## 📁 Repository Structure

```
├── app.py                # Main Gradio application code (Inference logic & UI)
├── requirements.txt      # Python dependencies
├── spaces.py             # Mock utility for HF Spaces zero-gpu decorators
├── assets/               # Image resources & font assets
│   ├── LXGWWenKai-Bold.ttf
│   ├── book.jpg
│   ├── ocr.jpg
│   ├── person.jpg
│   └── sweet.jpg
└── .gitignore            # Git exclusion rules
```

---

## 📝 Citation

If you find LocateAnything helpful in your research or applications, please cite:
```bibtex
@article{locateanything2026,
  title={LocateAnything: Locate Any Object in Images or Videos with Natural Language},
  author={NVIDIA Research},
  journal={arXiv preprint arXiv:2605.27365},
  year={2026}
}
```

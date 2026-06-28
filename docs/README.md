# 🧠 HQC-Vision

### Hybrid Quantum Computing Vision Framework for Industrial Inspection, Medical Imaging, and Hyperspectral Remote Sensing

## 📖 Abstract

HQC-Vision is a modular Hybrid Quantum Computing framework that integrates classical computer vision techniques with quantum search algorithms to solve complex image analysis problems. The framework demonstrates how Grover's Algorithm can be incorporated into vision pipelines for three real-world application domains:

- 🔧 Industrial Inspection (Blade Crack Detection)
- 🩺 Medical Imaging (Breast Microcalcification Detection)
- 🛰️ Hyperspectral Remote Sensing (Change Detection)

Built using Python, Qiskit, OpenCV, and scientific computing libraries, HQC-Vision provides a unified architecture for preprocessing, quantum feature encoding, Grover-based search, error mitigation, post-processing, and visualization.

## ✨ Key Features

- 🧠 Unified Hybrid Quantum Computing framework for computer vision applications
- ⚛️ Integration of Grover's Algorithm using Qiskit
- 🔧 Blade Crack Detection for industrial inspection
- 🩺 Breast Microcalcification Detection for medical imaging
- 🛰️ Hyperspectral Change Detection for remote sensing
- 🖼️ Image preprocessing and feature extraction pipeline
- 📊 Visualization and post-processing of detection results
- 📦 Modular architecture for extending to new vision applications

## 🛠 Technology Stack

| Category | Technologies |
|----------|--------------|
| **Programming Language** | Python 3.x |
| **Quantum Computing** | Qiskit |
| **Computer Vision** | OpenCV |
| **Scientific Computing** | NumPy, SciPy |
| **Machine Learning** | Scikit-learn |
| **Visualization** | Matplotlib |
| **Image Processing** | OpenCV, NumPy |
| **Configuration** | YAML |

## 🏗️ Framework Architecture

<p align="center">
  <img src="../images/HQC_Vision_Architecture.drawio.png"
       alt="HQC-Vision Architecture"
       width="900"/>
</p>

The HQC-Vision framework integrates three computer vision applications into a unified Hybrid Quantum Computing pipeline. The workflow begins with Image Preprocessing, followed by Feature Extraction, Quantum Feature Encoding, Grover Search Algorithm, Quantum Measurement, Error Mitigation, Post Processing, and finally Visualization & Detection Results.


## 📂 Repository Structure

```text
HQC-Vision
│
├── Blade Crack Detection/
│   ├── src/                  # Blade crack detection modules
│   ├── outputs/              # Detection results
│   ├── requirements.txt
│   ├── config.py
│   └── main.py
│
├── Hyperspectral Change Detection/
│   ├── modules/              # Quantum processing modules
│   ├── outputs/              # Generated change maps
│   ├── requirements.txt
│   ├── config.py
│   └── main.py
│
├── Microcalcification Detection/
│   ├── src/                  # Medical image processing modules
│   ├── outputs/              # Detection outputs
│   ├── dataset/              # Sample dataset structure
│   ├── requirements.txt
│   ├── config.yaml
│   └── main.py
│
├── docs/                     # Documentation
├── images/                   # Figures and screenshots
├── README.md
└── .gitignore
```

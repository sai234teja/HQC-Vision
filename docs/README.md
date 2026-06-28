# 🧠 HQC-Vision

> Hybrid Quantum Computing Vision Framework for Industrial Inspection, Medical Imaging, and Hyperspectral Remote Sensing

<p align="center">

![Python](https://img.shields.io/badge/Python-3.11-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-ComputerVision-green)
![Qiskit](https://img.shields.io/badge/Qiskit-Quantum-purple)
![License](https://img.shields.io/badge/License-MIT-orange)

</p>

---

## 📖 Abstract

HQC-Vision is a modular Hybrid Quantum Computing framework integrating classical Computer Vision with Quantum Search algorithms to solve image analysis problems efficiently.

The framework demonstrates how Grover's Algorithm can be incorporated into Computer Vision pipelines for three different real-world applications:

- 🔧 Industrial Blade Crack Detection
- 🩺 Medical Microcalcification Detection
- 🛰️ Hyperspectral Change Detection

The framework is designed to be modular, scalable, and adaptable to future quantum hardware.

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

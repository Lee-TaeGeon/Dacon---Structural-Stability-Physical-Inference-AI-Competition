# 🧠 Physical Reasoning AI (DACON)

> End-to-End pipeline from PyBullet-based data generation to deep learning-based physical stability prediction

---

## 📌 Overview

본 프로젝트는 구조물의 물리적 안정성(Stable / Unstable)을 예측하는 AI 모델을 개발하는 것을 목표로 합니다.
PyBullet을 활용하여 데이터를 직접 생성하고, 이를 기반으로 딥러닝 모델을 학습하여 물리 추론 문제를 해결했습니다.

---

## 🏆 Competition Result

* Public Leaderboard: **38th**
* Private Leaderboard: **28th**

> Private score가 더 높게 나타나 모델의 일반화 성능이 안정적으로 작동했음을 확인

---

## 🏗️ Data Generation (My Work)

* PyBullet 기반 물리 시뮬레이션으로 데이터 생성
* 다양한 구조 생성 (tower, pyramid, overhang 등)
* 시뮬레이션 결과를 활용한 자동 라벨링 (Stable / Unstable)
* 이미지 + JSON + CSV 형태의 데이터 구성

---

## 🤖 Model

### 🔹 Baseline (Team)

* Dual-branch architecture (front image + top view)
* ConvNeXt backbone (timm)
* Weighted sampler 및 TTA 적용
* 높은 성능 달성

> 본 프로젝트는 해당 baseline 모델을 기반으로 추가 실험을 진행

### 🔹 My Contribution

* 영상 프레임 기반 정보 추가 시도
* temporal feature를 활용한 성능 개선 실험

---

## 📊 Experiment & Insight

### ✔ Result

* Baseline 대비 성능 소폭 감소

### ✔ Insight

* temporal 정보가 항상 성능 향상을 보장하지 않음을 확인
* frame selection 및 noise가 성능에 영향을 줄 가능성 존재
* 단일 이미지 기반 feature가 더 효과적인 경우 확인

---

## 📂 Project Structure

```
dacon-physical-reasoning/
├── data_generation/        # PyBullet 데이터 생성 코드
│   └── generate_dataset.py
├── notebooks/              # 모델 실험 및 학습 코드
│   └── video_experiment.ipynb
├── requirements.txt
└── README.md
```

---

## 🚀 How to Run

### 1️⃣ 데이터 생성

```bash
python data_generation/generate_dataset.py
```

### 2️⃣ 모델 실행

* `notebooks/video_experiment.ipynb` 실행

---

## 🧩 Tech Stack

* **Simulation**: PyBullet
* **Deep Learning**: PyTorch, timm (ConvNeXt)
* **Data Processing**: Pandas, NumPy
* **Computer Vision**: OpenCV
* **ML Utils**: scikit-learn

---

## 💡 Key Takeaway

* 데이터 생성부터 모델링까지 End-to-End 파이프라인 구축
* baseline 모델을 기반으로 실험 설계 및 성능 비교 수행
* temporal feature 활용의 한계와 데이터 품질의 중요성 확인
* 일반화 성능이 안정적인 모델 설계 경험 확보

---

## 🙋‍♂️ Author

* Lee TaeGeon

  * Data Generation (PyBullet)
  * Experiment & Analysis
* Baseline Model: Team Project

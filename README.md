# Focus Drive

Focus Drive는 운전 중 졸음 상태를 실시간으로 감지하고, 경고음을 통해 운전자에게 위험을 알리는 모니터링 시스템이다.  
OpenCV를 사용해 웹캠으로부터 프레임 이미지를 수집하고, Google Cloud Vertex AI AutoML Vision을 통해 졸음 여부를 분류한다.

---

## 프로젝트 개요

운전 중 졸음은 교통사고의 주요 원인 중 하나이다.  
Focus Drive는 졸음 운전 사고를 예방하고자, 운전자의 눈 상태를 기반으로 실시간 졸음 여부를 판단하고 즉시 경고음을 발생시킨다.

- 웹캠 영상 수집 및 실시간 처리
- GCP AutoML Vision으로 졸음 상태 분류
- 졸음 감지 시 경고음 출력

---

## 시연 영상 및 서비스 화면

Focus Drive가 실시간으로 작동하는 모습은 아래 영상과 이미지로 확인할 수 있다.

- [시연 영상 보기](https://youtu.be/KYvnuyHBH5s)

<img width="800" alt="서비스 실행 화면 캡처" src="https://github.com/user-attachments/assets/105bd643-4d83-4d26-8349-097687b6afe0" />

---

## 기술 스택

- Python 3.10
- OpenCV: 실시간 프레임 수집
- Google Cloud Platform (Vertex AI AutoML Vision, Cloud Storage)
- pygame: 경고음 출력
- requests: Vertex AI endpoint 호출

---

## 시스템 아키텍처

<img width="800" alt="시스템 아키텍처" src="https://github.com/user-attachments/assets/beb8db54-e4e1-45ee-b144-d66ad0ddc197" />

1. OpenCV로 실시간 프레임을 수집한다.  
2. Vertex AI AutoML Vision에 이미지를 전송해 졸음 여부를 예측한다.  
3. 예측 결과가 'DROWSY'일 경우 pygame으로 경고음을 출력한다.

---

## 주요 기능

| 기능명       | 설명 |
|--------------|------|
| 졸음 탐지     | 실시간으로 운전자 얼굴 이미지를 분류 (NATURAL / DROWSY) |
| 경고 시스템   | 졸음 감지 시 즉각 경고음을 출력 |
| 실시간 처리   | 프레임 수집 → 예측 → 반응까지 1초 이내 처리 |

---

## GCP 활용 모델 학습 및 배포

Focus Drive는 GCP Vertex AI AutoML Vision을 활용해 졸음 여부를 분류하는 모델을 사용한다.  
초기 실행을 위해 사용자는 아래 과정을 따라 직접 모델을 학습하고 배포해야 한다.

자세한 설명은 아래 블로그에서 확인할 수 있다:

- [데이터셋 업로드 및 모델 학습](https://chohaeminn.tistory.com/21)  
- [모델 배포 및 API 키 설정](https://chohaeminn.tistory.com/22)

---

### 준비 절차 요약

1. GCP 콘솔에서 AutoML Vision 데이터셋 생성
2. 이미지 업로드 및 라벨 지정 (`NATURAL`, `DROWSY`)
3. 모델 학습 후 Vertex AI에서 배포 (endpoint 생성)
4. endpoint ID를 확보하여 코드에 반영
5. 서비스 계정 생성 및 `.json` 형식의 API 키 발급
6. 해당 키 파일을 프로젝트 디렉토리에 위치시킴

---

### 실행 준비 방법

- `.json` API 키 파일을 프로젝트 폴더에 두고, 코드에서 이를 참조한다.
- `endpoint` 값은 사용자가 배포한 실제 모델의 ID로 수정한다:

```python
endpoint = "projects/your-project-id/locations/us-central1/endpoints/XXXXXXXXXXXXXXX"
```
- 모든 설정이 완료되면 아래 명령어로 실행할 수 있다:

```bash
python realtime_drowsy.py
```

---

### 실행방법

1. 가상환경 생성 및 패키지 설치

```bash
conda create -n focusdrive python=3.10
conda activate focusdrive
pip install -r requirements.txt
```

2. 실시간 감지 시스템 실행

```
python realtime_drowsy.py
```

---

### 프로젝트 구조

```bash
FocusDrive/
├── realtime_drowsy.py      # 실시간 졸음 감지 실행 파일
├── alarming.wav            # 경고음 오디오 파일
├── requirements.txt        # 의존성 목록
└── README.md
```
---

### 모델 학습 정보

  - 분류 클래스: NATURAL / DROWSY
  - 학습 플랫폼: Vertex AI AutoML Vision
  - 데이터셋: Kaggle Drowsiness Dataset, Yawn Video Dataset
  - 라벨링: 수동 라벨링 후 CSV 업로드
  - 검증 정확도: 약 94.1%

---

### 회고 및 개선 사항

  - **GCP 키 인증 오류**
    - 서비스 계정 키 파일을 등록했음에도 인증 오류가 반복되었고, API 호출이 계속 실패했다. 원인을 정확히 알기 어려웠고, 기존 프로젝트에서 설정이 꼬인 것으로 판단해 새 계정과 프로젝트로 전환했다.
    - 이 경험을 통해 GCP 인증은 단순히 키 파일만으로 해결되는 것이 아니라, IAM 역할 부여, API 사용 설정, 프로젝트 연동이 유기적으로 작동해야 한다는 점을 이해하게 되었다.
  
  - **AutoML 노드 제한으로 인한 배포 실패**
    - 모델 학습은 문제없이 완료되었으나, 배포 단계에서 `"AutoMLImageClassificationDeployedModelNodes exceeded"` 오류가 발생하며 배포에 실패했다.
    - GCP에서는 리소스가 기본적으로 매우 낮은 제한(Quota)으로 설정되어 있으며, 이를 사전에 파악하고 할당량을 관리하는 것이 중요하다는 점을 배웠다.
  
  - **크레딧 소진으로 인한 프로젝트 전환**
    - GCP 무료 크레딧이 빠르게 소진되어 모델 배포와 예측 테스트가 중단되었고, 최종적으로는 새 계정을 생성해 프로젝트를 재구성했다.
    - 실습/개발 단계에서도 비용과 리소스를 염두에 둔 설계가 필요하며, 클라우드 기반 프로젝트의 경우 요금 정책과 소모 구조를 이해하는 것이 매우 중요하다는 점을 깨달았다.

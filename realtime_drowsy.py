import os
import base64
import pygame
import cv2
from google.cloud import aiplatform

# 🟡 1. 환경 설정
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "focusdrive-466303-fe2a631b333b.json"
PROJECT_ID = "focusdrive-466303"
LOCATION = "us-central1"
ENDPOINT_ID = "1425597089760411648"

# 🟡 2. Vertex AI 초기화
aiplatform.init(project=PROJECT_ID, location=LOCATION)
endpoint = aiplatform.Endpoint(endpoint_name=ENDPOINT_ID)

# 🟡 3. 알람 사운드 초기화
pygame.mixer.init()
alarm_sound = pygame.mixer.Sound("alarming.wav")  # wav 파일이 같은 폴더에 있어야 함

# 🟢 4. 실시간 웹캠 감지 시작
print("🟢 실시간 졸음 감지가 시작됩니다.")
cap = cv2.VideoCapture(0)

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❗ 카메라에서 프레임을 읽을 수 없습니다.")
            break

        # ✅ 화면에 웹캠 이미지 띄우기
        cv2.imshow("Webcam", frame)

        # 5. 이미지 인코딩 → base64 변환
        _, buffer = cv2.imencode('.jpg', frame)
        image_bytes = buffer.tobytes()
        image_b64 = base64.b64encode(image_bytes).decode('utf-8')

        try:
            # 🧠 Vertex AI 모델 예측
            response = endpoint.predict(instances=[{"content": image_b64}])
            prediction = response.predictions[0]  # dict 형식 반환

            print("🔍 예측 결과:", prediction)

            # ✅ ids 없이 필요한 정보만 추출
            display_names = prediction['displayNames']
            confidences = prediction['confidences']

            # 🟠 DROWSY 확률이 0.5 이상일 경우 경고음 재생
            if 'DROWSY' in display_names:
                drowsy_index = display_names.index('DROWSY')
                drowsy_confidence = confidences[drowsy_index]

                if drowsy_confidence >= 0.5:
                    if not pygame.mixer.get_busy():  # 중복 재생 방지
                        alarm_sound.play()

        except Exception as e:
            print("❗ 예측 중 오류 발생:", e)

        # 🔚 ESC 키 누르면 종료
        if cv2.waitKey(1) & 0xFF == 27:
            break

finally:
    cap.release()
    cv2.destroyAllWindows()

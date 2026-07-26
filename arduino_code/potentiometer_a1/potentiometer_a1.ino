// A1 가변저항 입력 확인용 스케치
//
// 연결:
//   가변저항 한쪽 끝  -> 5V
//   가변저항 반대쪽  -> GND
//   가변저항 가운데  -> A1
//
// 시리얼 모니터 속도: 9600 baud

const uint8_t POTENTIOMETER_PIN = A1;
const unsigned long PRINT_INTERVAL_MS = 100;
const uint8_t SAMPLE_COUNT = 8;

unsigned long lastPrintTime = 0;

void setup() {
  pinMode(POTENTIOMETER_PIN, INPUT);
  Serial.begin(9600);
  delay(500);
  Serial.println("A1 rotation sensor monitor started");
}

void loop() {
  const unsigned long now = millis();
  if (now - lastPrintTime < PRINT_INTERVAL_MS) {
    return;
  }
  lastPrintTime = now;

  // 순간 노이즈를 줄이기 위해 8회 측정값의 평균을 사용한다.
  unsigned long sum = 0;
  for (uint8_t i = 0; i < SAMPLE_COUNT; ++i) {
    sum += analogRead(POTENTIOMETER_PIN);
  }

  const int rawValue = sum / SAMPLE_COUNT;
  const float voltage = rawValue * (5.0f / 1023.0f);
  const int percent = map(rawValue, 0, 1023, 0, 100);

  Serial.print("A1 raw=");
  Serial.print(rawValue);
  Serial.print(", voltage=");
  Serial.print(voltage, 2);
  Serial.print("V, percent=");
  Serial.print(percent);
  Serial.println("%");
}

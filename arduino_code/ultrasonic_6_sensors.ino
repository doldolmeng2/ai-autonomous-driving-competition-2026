/*
  ==========================================================
  미래차 경진대회 - 초음파 센서 6개 송신 코드
  ==========================================================

  ROS2 sensor_topic/ultrasonic_node.py와 맞는 Serial 출력 형식:

    "거리1,거리2,거리3,거리4,거리5,거리6\n"

  거리 단위:
    meter

  예:
    센서 6개: 0.153,0.842,nan,1.204,0.331,2.018
    센서 5개: 0.153,0.842,nan,1.204,0.331

  토픽 매핑:
    값 1 -> /ultrasonic/range_1
    값 2 -> /ultrasonic/range_2
    값 3 -> /ultrasonic/range_3
    값 4 -> /ultrasonic/range_4
    값 5 -> /ultrasonic/range_5
    값 6 -> /ultrasonic/range_6

  주의:
    Serial에는 CSV 거리값만 출력한다.
    디버그 문장을 출력하면 ROS 노드가 파싱하지 못할 수 있다.
*/

const int ACTIVE_SENSOR_COUNT = 6;  // 센서 5개만 쓸 때는 5로 변경

// // Arduino Mega 기준 기본 핀. 앞에서부터 ACTIVE_SENSOR_COUNT개만 사용함.
const int TRIG_PINS[] = {22, 24, 26, 28, 30, 32};
const int ECHO_PINS[] = {23, 25, 27, 29, 31, 33};

const int CONFIGURED_SENSOR_COUNT = sizeof(TRIG_PINS) / sizeof(TRIG_PINS[0]);
const int SENSOR_COUNT = min(ACTIVE_SENSOR_COUNT, CONFIGURED_SENSOR_COUNT);

const unsigned long BAUDRATE = 115200;

// 센서 하나당 최대 대기 시간. 25000us는 약 4.3m 정도.
const unsigned long ECHO_TIMEOUT_US = 25000;

// 전체 6개 값을 publish하는 주기.
const unsigned long PUBLISH_PERIOD_MS = 50;

// HC-SR04 계열 음속 변환값.
const float SOUND_SPEED_M_PER_US = 0.000343;

unsigned long lastPublishTime = 0;

void setup() {
  Serial.begin(BAUDRATE);

  for (int i = 0; i < SENSOR_COUNT; i++) {
    pinMode(TRIG_PINS[i], OUTPUT);
    pinMode(ECHO_PINS[i], INPUT);
    digitalWrite(TRIG_PINS[i], LOW);
  }

  delay(100);
}

void loop() {
  unsigned long now = millis();
  if (now - lastPublishTime < PUBLISH_PERIOD_MS) {
    return;
  }
  lastPublishTime = now;

  for (int i = 0; i < SENSOR_COUNT; i++) {
    float distanceM = readDistanceMeters(TRIG_PINS[i], ECHO_PINS[i]);

    if (i > 0) {
      Serial.print(',');
    }

    if (isnan(distanceM)) {
      Serial.print("nan");
    } else {
      Serial.print(distanceM, 3);
    }

    // 센서 간 간섭을 조금 줄이기 위한 짧은 간격.
    delay(5);
  }

  Serial.println();
}

float readDistanceMeters(int trigPin, int echoPin) {
  digitalWrite(trigPin, LOW);
  delayMicroseconds(2);

  digitalWrite(trigPin, HIGH);
  delayMicroseconds(10);
  digitalWrite(trigPin, LOW);

  unsigned long duration = pulseIn(echoPin, HIGH, ECHO_TIMEOUT_US);
  if (duration == 0) {
    return NAN;
  }

  // 왕복 시간이므로 2로 나눔.
  return (duration * SOUND_SPEED_M_PER_US) / 2.0;
}

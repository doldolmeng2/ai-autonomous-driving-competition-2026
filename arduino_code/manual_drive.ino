/*
  ==========================================================
  미래차 경진대회 - Arduino Mega 모터 제어 코드
  ==========================================================

  Serial Protocol:
    "steer drive\n"

  예:
    40 30
    -40 30
    0 0

  steer:
    양수 = 오른쪽 조향
    음수 = 왼쪽 조향
    0    = 조향 정지

  drive:
    양수 = 전진
    음수 = 후진
    0    = 구동 정지

  실제 analogWrite PWM 값:
    조향: 0~40
    속도: 0~30
*/

#define STEER_IN1   7
#define STEER_IN2   6

#define MOTOR1_IN1  13
#define MOTOR1_IN2  12

#define MOTOR2_IN1  11
#define MOTOR2_IN2  10

const int MAX_STEER_PWM = 40;
const int MAX_DRIVE_PWM = 30;

const unsigned long COMMAND_TIMEOUT_MS = 500;

#define DEBUG_SERIAL 0
#define DEBUG_EVERY_N 1

unsigned long lastCommandTime = 0;
bool isStoppedByTimeout = false;
int commandCount = 0;

void setup() {
  Serial.begin(115200);
  Serial.setTimeout(20);

  pinMode(STEER_IN1, OUTPUT);
  pinMode(STEER_IN2, OUTPUT);

  pinMode(MOTOR1_IN1, OUTPUT);
  pinMode(MOTOR1_IN2, OUTPUT);

  pinMode(MOTOR2_IN1, OUTPUT);
  pinMode(MOTOR2_IN2, OUTPUT);

  stopAll();
  lastCommandTime = millis();

#if DEBUG_SERIAL
  Serial.println("Arduino motor controller ready.");
  Serial.println("Serial protocol: steer drive");
  Serial.println("Baudrate: 115200");
  Serial.println("Max steer PWM: 40");
  Serial.println("Max drive PWM: 30");
#endif
}

void loop() {
  readSerialCommand();

  if (millis() - lastCommandTime > COMMAND_TIMEOUT_MS) {
    if (!isStoppedByTimeout) {
      stopAll();
      isStoppedByTimeout = true;

#if DEBUG_SERIAL
      Serial.println("[TIMEOUT] No command received. Motors stopped.");
#endif
    }
  }
}

void readSerialCommand() {
  if (!Serial.available()) {
    return;
  }

  String line = Serial.readStringUntil('\n');
  line.trim();

  if (line.length() == 0) {
    return;
  }

  int steer = 0;
  int drive = 0;
  int parsed = sscanf(line.c_str(), "%d %d", &steer, &drive);

  if (parsed == 2) {
    setSteer(steer);
    setDrive(drive);

    lastCommandTime = millis();
    isStoppedByTimeout = false;
    commandCount++;

#if DEBUG_SERIAL
    if (commandCount % DEBUG_EVERY_N == 0) {
      Serial.print("[RX] raw: ");
      Serial.print(line);
      Serial.print(" | steer: ");
      Serial.print(steer);
      Serial.print(" | drive: ");
      Serial.print(drive);
      Serial.print(" | steer_pwm: ");
      Serial.print(steerPwmMagnitude(steer));
      Serial.print(" | drive_pwm: ");
      Serial.println(drivePwmMagnitude(drive));
    }
#endif

  } else {
#if DEBUG_SERIAL
    Serial.print("[ERROR] Invalid command: ");
    Serial.println(line);
    Serial.println("Expected format: steer drive");
#endif
  }
}

int steerPwmMagnitude(int val) {
  return constrain(abs(val), 0, MAX_STEER_PWM);
}

int drivePwmMagnitude(int val) {
  return constrain(abs(val), 0, MAX_DRIVE_PWM);
}

void setDrive(int val) {
  int pwm = drivePwmMagnitude(val);

  if (val > 0) {
    analogWrite(MOTOR1_IN1, pwm);
    digitalWrite(MOTOR1_IN2, LOW);

    analogWrite(MOTOR2_IN1, pwm);
    digitalWrite(MOTOR2_IN2, LOW);

  } else if (val < 0) {
    digitalWrite(MOTOR1_IN1, LOW);
    analogWrite(MOTOR1_IN2, pwm);

    digitalWrite(MOTOR2_IN1, LOW);
    analogWrite(MOTOR2_IN2, pwm);

  } else {
    digitalWrite(MOTOR1_IN1, LOW);
    digitalWrite(MOTOR1_IN2, LOW);

    digitalWrite(MOTOR2_IN1, LOW);
    digitalWrite(MOTOR2_IN2, LOW);
  }
}

void setSteer(int val) {
  int pwm = steerPwmMagnitude(val);

  if (val > 0) {
    analogWrite(STEER_IN1, pwm);
    digitalWrite(STEER_IN2, LOW);

  } else if (val < 0) {
    digitalWrite(STEER_IN1, LOW);
    analogWrite(STEER_IN2, pwm);

  } else {
    digitalWrite(STEER_IN1, LOW);
    digitalWrite(STEER_IN2, LOW);
  }
}

void stopAll() {
  setSteer(0);
  setDrive(0);
}
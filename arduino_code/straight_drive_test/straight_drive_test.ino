/*
  Straight driving test for Arduino Mega 2560.

  On every power-up/reset:
    1. Keep steering and rear motors stopped for START_DELAY_MS.
    2. Drive both rear motors forward for DRIVE_TIME_MS.
    3. Stop permanently until the next reset.

  WARNING: Lift the vehicle or secure a clear test area before powering it.
*/

#include <Arduino.h>

constexpr uint8_t STEER_IN1 = 7;
constexpr uint8_t STEER_IN2 = 6;
constexpr uint8_t MOTOR1_IN1 = 5;
constexpr uint8_t MOTOR1_IN2 = 12;
constexpr uint8_t MOTOR2_IN1 = 11;
constexpr uint8_t MOTOR2_IN2 = 10;

// 실차에 맞게 이 값들을 조정한다.
constexpr int DRIVE_PWM = 80;                    // 1~130
constexpr unsigned long START_DELAY_MS = 3000;  // 작동 전 3초 대기
constexpr unsigned long DRIVE_TIME_MS = 1000;   // 1초 전진

void stopSteering() {
  digitalWrite(STEER_IN1, LOW);
  digitalWrite(STEER_IN2, LOW);
}

void stopRearMotors() {
  digitalWrite(MOTOR1_IN1, LOW);
  digitalWrite(MOTOR1_IN2, LOW);
  digitalWrite(MOTOR2_IN1, LOW);
  digitalWrite(MOTOR2_IN2, LOW);
}

void driveRearMotorsForward(int value) {
  const int pwm = constrain(value, 0, 130);

  analogWrite(MOTOR1_IN1, pwm);
  digitalWrite(MOTOR1_IN2, LOW);

  analogWrite(MOTOR2_IN1, pwm);
  digitalWrite(MOTOR2_IN2, LOW);
}

void setup() {
  pinMode(STEER_IN1, OUTPUT);
  pinMode(STEER_IN2, OUTPUT);
  pinMode(MOTOR1_IN1, OUTPUT);
  pinMode(MOTOR1_IN2, OUTPUT);
  pinMode(MOTOR2_IN1, OUTPUT);
  pinMode(MOTOR2_IN2, OUTPUT);

  stopSteering();
  stopRearMotors();

  delay(START_DELAY_MS);
  driveRearMotorsForward(DRIVE_PWM);
  delay(DRIVE_TIME_MS);
  stopRearMotors();
}

void loop() {
  // 한 번만 주행한다. 다시 시험하려면 Arduino의 RESET 버튼을 누른다.
  stopSteering();
  stopRearMotors();
  delay(100);
}

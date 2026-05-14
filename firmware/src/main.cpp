#include <Arduino.h>
#include <AccelStepper.h>
#include <Servo.h>

// CNC Shield V3 axis pins
#define X_STEP_PIN 2
#define X_DIR_PIN 5
#define Z_STEP_PIN 4
#define Z_DIR_PIN 7
#define ENABLE_PIN 8

// SG90 servo on Y_STEP pin (D3)
#define SERVO_PIN 3

// 28BYJ-48 via ULN2003 on analog pins
#define BYJ_IN1 A0
#define BYJ_IN2 A1
#define BYJ_IN3 A2
#define BYJ_IN4 A3

// Second 28BYJ-48 via ULN2003
#define BYJ2_IN1 11
#define BYJ2_IN2 12
#define BYJ2_IN3 A4
#define BYJ2_IN4 A5

// L298N DC motor (using motor A side: IN1, IN2, ENA)
#define L298N_IN1 9
#define L298N_IN2 10
#define L298N_ENA 6  // PWM for speed control

// X-axis NEMA stepper (A4988 driver)
AccelStepper xStepper(AccelStepper::DRIVER, X_STEP_PIN, X_DIR_PIN);

// Z-axis NEMA stepper (A4988 driver)
AccelStepper zStepper(AccelStepper::DRIVER, Z_STEP_PIN, Z_DIR_PIN);

// 28BYJ-48 (half-step mode, note pin order: IN1, IN3, IN2, IN4)
AccelStepper byjStepper(AccelStepper::HALF4WIRE, BYJ_IN1, BYJ_IN3, BYJ_IN2, BYJ_IN4);

// Second 28BYJ-48 (half-step mode)
AccelStepper byj2Stepper(AccelStepper::HALF4WIRE, BYJ2_IN1, BYJ2_IN3, BYJ2_IN2, BYJ2_IN4);

Servo gripServo;

String inputBuffer = "";

void setup() {
  Serial.begin(115200);

  // Enable stepper drivers (active LOW)
  pinMode(ENABLE_PIN, OUTPUT);
  digitalWrite(ENABLE_PIN, LOW);

  xStepper.setMaxSpeed(1000);
  xStepper.setAcceleration(500);

  zStepper.setMaxSpeed(1000);
  zStepper.setAcceleration(500);

  byjStepper.setMaxSpeed(800);
  byjStepper.setAcceleration(400);

  byj2Stepper.setMaxSpeed(800);
  byj2Stepper.setAcceleration(400);

  gripServo.attach(SERVO_PIN);
  gripServo.write(0);

  // L298N pins
  pinMode(L298N_IN1, OUTPUT);
  pinMode(L298N_IN2, OUTPUT);
  pinMode(L298N_ENA, OUTPUT);
  digitalWrite(L298N_IN1, LOW);
  digitalWrite(L298N_IN2, LOW);
  analogWrite(L298N_ENA, 0);

  Serial.println("Control Ready");
  Serial.println("Commands: X<steps> Z<steps> B<steps> J<steps> O<angle> F<speed> G<speed> S P");
}

void processCommand(String cmd) {
  cmd.trim();
  if (cmd.length() == 0) return;

  char command = toupper(cmd.charAt(0));
  long value = 0;
  if (cmd.length() > 1) {
    value = cmd.substring(1).toInt();
  }

  switch (command) {
    case 'X': // X-axis NEMA stepper
      xStepper.move(value);
      Serial.print("X MOVE ");
      Serial.print(value);
      Serial.print(" -> ");
      Serial.println(xStepper.targetPosition());
      break;

    case 'Z': // Z-axis NEMA stepper
      zStepper.move(value);
      Serial.print("Z MOVE ");
      Serial.print(value);
      Serial.print(" -> ");
      Serial.println(zStepper.targetPosition());
      break;

    case 'B': // 28BYJ-48 #1
      byjStepper.move(value);
      Serial.print("BYJ1 MOVE ");
      Serial.print(value);
      Serial.print(" -> ");
      Serial.println(byjStepper.targetPosition());
      break;

    case 'J': // 28BYJ-48 #2
      byj2Stepper.move(value);
      Serial.print("BYJ2 MOVE ");
      Serial.print(value);
      Serial.print(" -> ");
      Serial.println(byj2Stepper.targetPosition());
      break;

    case 'O': // Servo angle (0-180)
      value = constrain(value, 0, 180);
      gripServo.write(value);
      Serial.print("SERVO -> ");
      Serial.println(value);
      break;

    case 'F': // DC motor forward
      value = constrain(value, 0, 100);
      digitalWrite(L298N_IN1, HIGH);
      digitalWrite(L298N_IN2, LOW);
      analogWrite(L298N_ENA, value);
      Serial.print("DC FWD speed: ");
      Serial.println(value);
      break;

    case 'G': // DC motor reverse
      value = constrain(value, 0, 100);
      digitalWrite(L298N_IN1, LOW);
      digitalWrite(L298N_IN2, HIGH);
      analogWrite(L298N_ENA, value);
      Serial.print("DC REV speed: ");
      Serial.println(value);
      break;

    case 'S': // DC motor stop
      digitalWrite(L298N_IN1, LOW);
      digitalWrite(L298N_IN2, LOW);
      analogWrite(L298N_ENA, 0);
      Serial.println("DC STOP");
      break;

    case 'R': // Reset all positions and servo
      xStepper.setCurrentPosition(0);
      zStepper.setCurrentPosition(0);
      byjStepper.setCurrentPosition(0);
      byj2Stepper.setCurrentPosition(0);
      gripServo.write(0);
      Serial.println("RESET ALL");
      break;

    case 'H': // Reset stepper positions only
      xStepper.setCurrentPosition(0);
      zStepper.setCurrentPosition(0);
      byjStepper.setCurrentPosition(0);
      byj2Stepper.setCurrentPosition(0);
      Serial.println("RESET STEPPERS");
      break;

    case 'P': // Print positions
      Serial.print("X: ");
      Serial.print(xStepper.currentPosition());
      Serial.print(" | Z: ");
      Serial.print(zStepper.currentPosition());
      Serial.print(" | BYJ1: ");
      Serial.print(byjStepper.currentPosition());
      Serial.print(" | BYJ2: ");
      Serial.println(byj2Stepper.currentPosition());
      break;

    default:
      Serial.print("Unknown: ");
      Serial.println(cmd);
      break;
  }
}

void loop() {
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\n' || c == '\r') {
      if (inputBuffer.length() > 0) {
        processCommand(inputBuffer);
        inputBuffer = "";
      }
    } else {
      inputBuffer += c;
    }
  }

  xStepper.run();
  zStepper.run();
  byjStepper.run();
  byj2Stepper.run();
}

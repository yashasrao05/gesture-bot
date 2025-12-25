#define IN1 5
#define IN2 6
#define IN3 9
#define IN4 10

#define SPEED 235
#define SPEED2 255  

void stopMotors() {
  analogWrite(IN1, 0);
  analogWrite(IN2, 0);
  analogWrite(IN3, 0);
  analogWrite(IN4, 0);
}

void forward() {
  analogWrite(IN1, 0);
  analogWrite(IN2, SPEED);
  analogWrite(IN3, SPEED);
  analogWrite(IN4, 0);
}

void backward() {
  analogWrite(IN1, SPEED);
  analogWrite(IN2, 0);
  analogWrite(IN3, 0);
  analogWrite(IN4, SPEED);
}

void left() {
  analogWrite(IN1, SPEED2);
  analogWrite(IN2, 0);
  analogWrite(IN3, SPEED2);
  analogWrite(IN4, 0);
}

void right() {
  analogWrite(IN1, 0);
  analogWrite(IN2, SPEED2);
  analogWrite(IN3, 0);
  analogWrite(IN4, SPEED2);
}

void setup() {
  pinMode(IN1, OUTPUT);
  pinMode(IN2, OUTPUT);
  pinMode(IN3, OUTPUT);
  pinMode(IN4, OUTPUT);

  stopMotors();

  Serial.begin(115200);
  Serial.println("Arduino Ready");
}

void loop() {
  if (Serial.available()) {
    char cmd = Serial.read();

    switch (cmd) {
      case 'F': forward();  break;
      case 'B': backward(); break;
      case 'L': left();     break;
      case 'R': right();    break;
      case 'S': stopMotors(); break;
      default: break;
    }
  }
}

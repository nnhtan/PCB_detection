#include <Arduino.h>
#include <ESP32Servo.h>

// ==========================================
// 1. CẤU HÌNH CHÂN GPIO
// ==========================================
// Động cơ bước (Driver TB6600 / DM542)
const int PIN_STEP = 5;  // Chân xung PUL+ (GPIO 5)
const int PIN_DIR  = 19; // Chân chiều DIR+ (GPIO 19)
const int PIN_ENA  = 18; // Chân ENA+ (LOW = Bật motor, HIGH = Ngắt lực)

// 2 CẢM BIẾN TIỆM CẬN (Mức LOW khi phát hiện phôi)
const int PIN_SENSOR_1 = 17; // Cảm biến 1: Vị trí cấp phôi ban đầu (GPIO 17)
const int PIN_SENSOR_2 = 16; // Cảm biến 2: Vị trí dừng băng tải 3s & gạt phôi (GPIO 16)
const int SENSOR_DETECT_LEVEL = LOW; // Mức tích cực khi thấy phôi (NPN = LOW)

// 2 SERVO SG90 (360 độ: 90 là DỪNG, 0 là QUAY GẠT)
const int PIN_SERVO_1 = 33; // Chân PWM điều khiển Servo 1 (GPIO 33)
const int PIN_SERVO_2 = 32; // Chân PWM điều khiển Servo 2 (GPIO 32)

// ==========================================
// 2. THÔNG SỐ ĐIỀU KHIỂN SERVO SG90 & THỜI GIAN
// ==========================================
// Đối với servo 360 độ:
const int SERVO_IDLE_ANGLE = 90; // 90: DỪNG HOÀN TOÀN (vị trí nghỉ)
const int SERVO_KICK_ANGLE = 0;  // 0 : Quay gạt phôi
const unsigned long SERVO_KICK_TIME_MS = 600; // Thời gian quay gạt phôi (ms)

// Đối tượng Servo
Servo servo1;
Servo servo2;

// Biến trạng thái non-blocking cho Servo 1
bool isServo1Kicking = false;
unsigned long servo1StartMs = 0;
int lastSensor1Val = HIGH;

// Biến trạng thái non-blocking cho Servo 2
bool isServo2Kicking = false;
unsigned long servo2StartMs = 0;
int lastSensor2Val = HIGH;

// ==========================================
// 3. THÔNG SỐ VẬN HÀNH ĐỘNG CƠ BƯỚC
// ==========================================
const unsigned long RAMP_UP_TIME_MS = 500;  // Thời gian tăng tốc (0.5 giây)

// THỜI GIAN DỪNG BĂNG TẢI KHI PHÁT HIỆN PHÔI: 3 GIÂY (3000ms)
const unsigned long STOP_TIME_MS = 3000;

// Tốc độ (Xung/giây - Steps Per Second):
// Cấu hình phù hợp với S1,S2,S3 = OFF, ON, OFF (Vi bước 1/8 - 1600 xung/vòng):
const float START_SPEED_SPS = 300.0;  // Tốc độ khởi động (êm ái)
const float MAX_SPEED_SPS   = 1000.0; // Tốc độ chạy chính (quay mượt, không rung)

// ==========================================
// 4. MÁY TRẠNG THÁI (FINITE STATE MACHINE) ĐỘNG CƠ BƯỚC
// ==========================================
enum SystemState {
  STATE_ACCEL,         // Giai đoạn 1: Chạy tăng tốc từ START lên MAX
  STATE_RUN_MAX_SPEED, // Giai đoạn 2: Duy trì tốc độ Max và chạy liên tục
  STATE_PAUSE_3S       // Giai đoạn 3: DỪNG BĂNG TẢI 3 GIÂY KHI CẢM BIẾN 2 THẤY PHÔI
};

SystemState currentState = STATE_ACCEL;

// Biến quản lý thời gian động cơ bước
unsigned long stateStartTimeMs = 0;
unsigned long lastStepUs = 0;
float currentSpeedSps = START_SPEED_SPS;

// Hàm phát 1 xung bước cho TB6600 (Non-blocking)
void triggerStep(float speedSps) {
  if (speedSps <= 0) return;
  unsigned long intervalUs = (unsigned long)(1000000.0 / speedSps);
  unsigned long nowUs = micros();

  if (nowUs - lastStepUs >= intervalUs) {
    lastStepUs = nowUs;
    digitalWrite(PIN_STEP, HIGH);
    delayMicroseconds(10); // Xung 10us đảm bảo Opto TB6600 nhận tốt tín hiệu
    digitalWrite(PIN_STEP, LOW);
  }
}

// ==========================================
// 5. HÀM QUẢN LÝ 2 CẢM BIẾN VÀ 2 SERVO (NON-BLOCKING)
// ==========================================
void updateServosAndSensors() {
  unsigned long now = millis();

  // ----------------------------------------
  // NHÁNH 1: Cảm biến 1 -> Servo 1 gạt phôi
  // ----------------------------------------
  int s1 = digitalRead(PIN_SENSOR_1);
  if (s1 == SENSOR_DETECT_LEVEL && lastSensor1Val != SENSOR_DETECT_LEVEL) {
    if (!isServo1Kicking && (now - servo1StartMs > 1500)) {
      Serial.println("\n[PHÂN LOẠI 1] Cảm biến 1 phát hiện phôi -> Servo 1 GẠT PHÔI!");
      servo1.write(SERVO_KICK_ANGLE);
      isServo1Kicking = true;
      servo1StartMs = now;
    }
  }
  lastSensor1Val = s1;

  // Hết thời gian gạt -> Dừng Servo 1 (về 90 độ)
  if (isServo1Kicking && (now - servo1StartMs >= SERVO_KICK_TIME_MS)) {
    servo1.write(SERVO_IDLE_ANGLE); // Dừng servo 360 độ
    isServo1Kicking = false;
    Serial.println("[PHÂN LOẠI 1] Servo 1 đã thu về vị trí nghỉ (DỪNG).");
  }

  // ----------------------------------------
  // NHÁNH 2: Cảm biến 2 -> DỪNG BĂNG TẢI 3S + Servo 2 gạt phôi
  // ----------------------------------------
  int s2 = digitalRead(PIN_SENSOR_2);
  if (s2 == SENSOR_DETECT_LEVEL && lastSensor2Val != SENSOR_DETECT_LEVEL) {
    // Tránh kích hoạt lặp nếu đang trong 3s dừng
    if (!isServo2Kicking && (now - servo2StartMs > (STOP_TIME_MS + 1000))) {
      Serial.println("\n>>> [PHÂN LOẠI 2] Cảm biến 2 phát hiện phôi! -> Servo 2 GẠT PHÔI!");
      Serial.println(">>> [BĂNG TẢI] DỪNG BĂNG TẢI 3 GIÂY...");
      
      // Kích hoạt Servo 2 gạt phôi
      servo2.write(SERVO_KICK_ANGLE);
      isServo2Kicking = true;
      servo2StartMs = now;

      // RA LỆNH CHO BĂNG TẢI DỪNG LẬP TỨC TRONG 3 GIÂY
      currentState = STATE_PAUSE_3S;
      stateStartTimeMs = now;
    }
  }
  lastSensor2Val = s2;

  // Hết thời gian gạt -> Dừng Servo 2 (về 90 độ)
  if (isServo2Kicking && (now - servo2StartMs >= SERVO_KICK_TIME_MS)) {
    servo2.write(SERVO_IDLE_ANGLE); // Dừng servo 360 độ
    isServo2Kicking = false;
    Serial.println("[PHÂN LOẠI 2] Servo 2 đã thu về vị trí nghỉ (DỪNG).");
  }
}

// ==========================================
// 6. SETUP
// ==========================================
void setup() {
  Serial.begin(115200);
  delay(500);

  Serial.println("\n=======================================================");
  Serial.println("  HỆ THỐNG ĐIỀU KHIỂN BĂNG TẢI + PHÂN LOẠI 2 SERVO");
  Serial.println("  - Cảm biến 2 phát hiện -> DỪNG BĂNG TẢI 3 GIÂY + Servo 2 gạt");
  Serial.println("  - Hết 3 giây dừng      -> Băng tải TỰ ĐỘNG CHẠY TIẾP");
  Serial.println("=======================================================");

  // 1. Cấu hình chân Động cơ bước TB6600
  pinMode(PIN_STEP, OUTPUT);
  pinMode(PIN_DIR,  OUTPUT);
  pinMode(PIN_ENA,  OUTPUT);
  digitalWrite(PIN_DIR, HIGH); // Chiều quay thuận
  digitalWrite(PIN_ENA, LOW);  // Kích hoạt driver (LOW = Enable)

  // 2. Cấu hình chân 2 Cảm biến tiệm cận
  pinMode(PIN_SENSOR_1, INPUT_PULLUP);
  pinMode(PIN_SENSOR_2, INPUT_PULLUP);

  lastSensor1Val = digitalRead(PIN_SENSOR_1);
  lastSensor2Val = digitalRead(PIN_SENSOR_2);

  // 3. Cấu hình 2 Servo SG90 360 độ
  ESP32PWM::allocateTimer(0);
  ESP32PWM::allocateTimer(1);
  ESP32PWM::allocateTimer(2);
  ESP32PWM::allocateTimer(3);

  servo1.setPeriodHertz(50);
  servo1.write(SERVO_IDLE_ANGLE); // 90 độ = DỪNG
  servo1.attach(PIN_SERVO_1, 1000, 2000);

  servo2.setPeriodHertz(50);
  servo2.write(SERVO_IDLE_ANGLE); // 90 độ = DỪNG
  servo2.attach(PIN_SERVO_2, 1000, 2000);

  delay(500);

  // Bắt đầu chu trình động cơ bước: Tăng tốc
  stateStartTimeMs = millis();
  currentState = STATE_ACCEL;
  Serial.println("[BĂNG TẢI] Bắt đầu chạy và tăng tốc...");
}

// ==========================================
// 7. LOOP CHÍNH
// ==========================================
void loop() {
  // 1. Quét cảm biến và điều khiển 2 Servo phân loại liên tục (Non-blocking)
  updateServosAndSensors();

  // 2. Máy trạng thái điều khiển động cơ bước
  unsigned long currentMillis = millis();

  switch (currentState) {

    // Giai đoạn 1: Tăng tốc tuyến tính từ START lên MAX (500ms)
    case STATE_ACCEL: {
      unsigned long elapsed = currentMillis - stateStartTimeMs;
      if (elapsed < RAMP_UP_TIME_MS) {
        float progress = (float)elapsed / (float)RAMP_UP_TIME_MS;
        currentSpeedSps = START_SPEED_SPS + progress * (MAX_SPEED_SPS - START_SPEED_SPS);
        triggerStep(currentSpeedSps);
      } else {
        currentSpeedSps = MAX_SPEED_SPS;
        currentState = STATE_RUN_MAX_SPEED;
        Serial.printf("[BĂNG TẢI] Đạt tốc độ MAX (%.0f SPS)! Đang chạy và quét phôi...\n", MAX_SPEED_SPS);
      }
      break;
    }

    // Giai đoạn 2: Duy trì tốc độ Max và chạy bình thường
    case STATE_RUN_MAX_SPEED: {
      triggerStep(MAX_SPEED_SPS);
      break;
    }

    // Giai đoạn 3: DỪNG BĂNG TẢI 3 GIÂY KHI CẢM BIẾN 2 PHÁT HIỆN PHÔI
    case STATE_PAUSE_3S: {
      // KHÔNG gọi triggerStep() -> ĐỘNG CƠ DỪNG HOÀN TOÀN TRONG 3 GIÂY
      if (currentMillis - stateStartTimeMs >= STOP_TIME_MS) {
        Serial.println("\n[BĂNG TẢI] ĐÃ HẾT 3 GIÂY DỪNG! Băng tải TIẾP TỤC CHẠY TIẾP...\n");
        currentState = STATE_RUN_MAX_SPEED; // Tiếp tục chạy lại tốc độ MAX
      }
      break;
    }
  }
}
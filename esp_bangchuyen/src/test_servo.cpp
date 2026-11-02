#include <Arduino.h>
#include <ESP32Servo.h>

// =============================================================================
// CẤU HÌNH PHẦN CỨNG
// =============================================================================
const int PIN_SERVO_1   = 33; // Servo 1: Cấp phôi (GPIO 33)
const int PIN_SERVO_2   = 32; // Servo 2: Gạt phôi (GPIO 32)

const int PIN_BTN_START = 25; // Nút START (GPIO 25)
const int PIN_BTN_STOP  = 26; // Nút STOP  (GPIO 26)

// Mức tích cực khi ấn nút:
// - Nút START: Thường mở (NO) nối GND -> Bình thường = HIGH, Khi ẤN = LOW
const int BTN_START_PRESSED = LOW;

// - Nút STOP : Thường đóng (NC) nối GND -> Bình thường = LOW, Khi ẤN = HIGH
// (Nếu dùng nút STOP loại thường mở NO thì đổi thành LOW)
const int BTN_STOP_PRESSED  = HIGH;

// =============================================================================
// THÔNG SỐ XUNG PWM ĐIỀU KHIỂN TỐC ĐỘ VÀ CHIỀU QUAY (SERVO 360 ĐỘ)
// =============================================================================
// Tần số PWM chuẩn: 50 Hz (1 xung = 20ms)
// - 1500us: Dừng hoàn toàn
// - Càng xa 1500us thì quay càng nhanh (1000us và 2000us là tốc độ tối đa)
const int PULSE_STOP = 1500; // Dừng
const int PULSE_FWD  = 1400; // Quay thuận (tốc độ êm, tránh sụt áp gây reset)
const int PULSE_REV  = 1600; // Quay nghịch (tốc độ êm, tránh sụt áp gây reset)

Servo servo1;
Servo servo2;

// =============================================================================
// BIẾN QUẢN LÝ TRẠNG THÁI & ĐẾM XUNG
// =============================================================================
// motorState: 0 = DỪNG, 1 = QUAY THUẬN, 2 = QUAY NGHỊCH
int motorState = 0;              
unsigned long pulseCount = 0;    // Biến đếm tổng số xung đã phát
unsigned long lastPulseTime = 0; // Quản lý chu kỳ 20ms

// Biến chống rung nút bấm (Debounce)
int lastStartVal = HIGH;
int startState   = HIGH;
unsigned long lastStartDebounce = 0;

int lastStopVal  = LOW;
int stopState    = LOW;
unsigned long lastStopDebounce  = 0;

const unsigned long DEBOUNCE_TIME = 50;

// =============================================================================
// HÀM DỪNG (STOP)
// =============================================================================
void stopServo() {
  motorState = 0;
  servo1.writeMicroseconds(PULSE_STOP);
  servo2.writeMicroseconds(PULSE_STOP);

  // In tổng số xung đã phát lên Serial
  Serial.println("\n=======================================================");
  Serial.print(">>> [STOP] ĐÃ DỪNG SERVO! Tổng số xung đã phát: ");
  Serial.print(pulseCount);
  Serial.printf(" xung (= %lu ms)\n", pulseCount * 20);
  
  // Đưa biến xung về 0
  pulseCount = 0;
  Serial.println(">>> ĐÃ ĐƯA BIẾN XUNG VỀ 0.");
  Serial.println(">>> Lần ấn START tiếp theo sẽ bắt đầu lại từ: QUAY THUẬN.");
  Serial.println("=======================================================\n");
}

// =============================================================================
// HÀM XỬ LÝ NÚT START: ĐẢO CHIỀU LUÂN PHIÊN (THUẬN <-> NGHỊCH)
// =============================================================================
void handleStartToggle() {
  // 1. Nếu đang DỪNG hoặc đang QUAY NGHỊCH -> Chuyển sang QUAY THUẬN
  if (motorState == 0 || motorState == 2) {
    if (motorState == 2) {
      // Dừng đệm 100ms triệt tiêu dòng hãm chống sụt nguồn Brownout
      servo1.writeMicroseconds(PULSE_STOP);
      servo2.writeMicroseconds(PULSE_STOP);
      delay(100);
    }
    motorState = 1;
    servo1.writeMicroseconds(PULSE_FWD);
    servo2.writeMicroseconds(PULSE_FWD);

    Serial.println("\n-------------------------------------------------------");
    Serial.println(">>> [NÚT START]: ĐẢO CHIỀU -> QUAY THUẬN");
    Serial.println("-------------------------------------------------------");
  }
  // 2. Nếu đang QUAY THUẬN -> Chuyển sang QUAY NGHỊCH
  else if (motorState == 1) {
    // Dừng đệm 100ms triệt tiêu dòng hãm chống sụt nguồn Brownout
    servo1.writeMicroseconds(PULSE_STOP);
    servo2.writeMicroseconds(PULSE_STOP);
    delay(100);

    motorState = 2;
    servo1.writeMicroseconds(PULSE_REV);
    servo2.writeMicroseconds(PULSE_REV);

    Serial.println("\n-------------------------------------------------------");
    Serial.println(">>> [NÚT START]: ĐẢO CHIỀU -> QUAY NGHỊCH");
    Serial.println("-------------------------------------------------------");
  }
}

// =============================================================================
// SETUP
// =============================================================================
void setup() {
  Serial.begin(115200);

  pinMode(PIN_BTN_START, INPUT_PULLUP);
  pinMode(PIN_BTN_STOP,  INPUT_PULLUP);

  lastStartVal = digitalRead(PIN_BTN_START);
  startState   = lastStartVal;
  lastStopVal  = digitalRead(PIN_BTN_STOP);
  stopState    = lastStopVal;

  ESP32PWM::allocateTimer(0);
  ESP32PWM::allocateTimer(1);
  ESP32PWM::allocateTimer(2);
  ESP32PWM::allocateTimer(3);

  servo1.setPeriodHertz(50);
  servo1.attach(PIN_SERVO_1, 1000, 2000);
  servo1.writeMicroseconds(PULSE_STOP);

  servo2.setPeriodHertz(50);
  servo2.attach(PIN_SERVO_2, 1000, 2000);
  servo2.writeMicroseconds(PULSE_STOP);

  Serial.println("\n=======================================================");
  Serial.println("  CHƯƠNG TRÌNH TEST HÀNH TRÌNH SERVO SG90 360 ĐỘ");
  Serial.println("  - Ấn nút START : Đảo chiều luân phiên (Thuận <-> Nghịch)");
  Serial.println("  - Ấn nút STOP  : Dừng servo và reset số xung về 0");
  Serial.println("  (Có thể gửi phím '1' = START, '0' = STOP qua Serial)");
  Serial.println("=======================================================\n");
}

// =============================================================================
// LOOP
// =============================================================================
void loop() {
  unsigned long now = millis();

  // 1. Nhận lệnh nhanh qua Serial Monitor
  if (Serial.available() > 0) {
    char c = Serial.read();
    if (c == '1' || c == ' ') handleStartToggle();
    else if (c == '0' || c == 's' || c == 'S') stopServo();
  }

  // 2. Đọc nút START (chống rung 50ms)
  int rStart = digitalRead(PIN_BTN_START);
  if (rStart != lastStartVal) lastStartDebounce = now;
  if ((now - lastStartDebounce) > DEBOUNCE_TIME) {
    if (rStart != startState) {
      startState = rStart;
      if (startState == BTN_START_PRESSED) {
        handleStartToggle();
      }
    }
  }
  lastStartVal = rStart;

  // 3. Đọc nút STOP (chống rung 50ms)
  int rStop = digitalRead(PIN_BTN_STOP);
  if (rStop != lastStopVal) lastStopDebounce = now;
  if ((now - lastStopDebounce) > DEBOUNCE_TIME) {
    if (rStop != stopState) {
      stopState = rStop;
      if (stopState == BTN_STOP_PRESSED) {
        stopServo();
      }
    }
  }
  lastStopVal = rStop;

  // 4. Phát và đếm xung liên tục khi motor đang quay (mỗi 20ms = 1 xung PWM)
  if (motorState != 0) {
    if (now - lastPulseTime >= 20) {
      lastPulseTime = now;
      pulseCount++;

      // Hiện số xung đã phát lên Serial
      if (motorState == 1) {
        Serial.printf("  [THUẬN]  Số xung: %4lu  (%6lu ms)\n", pulseCount, pulseCount * 20);
      } else {
        Serial.printf("  [NGHỊCH] Số xung: %4lu  (%6lu ms)\n", pulseCount, pulseCount * 20);
      }
    }
  }
}

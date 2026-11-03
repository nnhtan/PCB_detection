#include <Arduino.h>
#include <ESP32Servo.h>

// =============================================================================
// 1. CẤU HÌNH CHÂN GPIO (GIỮ NGUYÊN 100% THEO PHẦN CỨNG)
// =============================================================================
const int PIN_STEP = 5;  // Xung PUL+ (GPIO 5)
const int PIN_DIR  = 19; // Chiều DIR+ (GPIO 19)
const int PIN_ENA  = 18; // Kích hoạt ENA+ (GPIO 18)

const int DIR_FORWARD = LOW; // Chiều chạy băng tải

// Nút nhấn
const int PIN_BTN_START = 25; // Nút START (GPIO 25)
const int PIN_BTN_STOP  = 26; // Nút STOP  (GPIO 26)

// 2 Cảm biến tiệm cận (Mức LOW khi thấy phôi)
const int PIN_SENSOR_1 = 16; // Cảm biến 1: CAMERA (GPIO 16)
const int PIN_SENSOR_2 = 17; // Cảm biến 2: CUỐI BĂNG TẢI (GPIO 17)
const int SENSOR_DETECT_LEVEL = LOW; 

// 2 Servo SG90 360 độ (Đã có PCB phụ cấp 5V)
const int PIN_SERVO_1 = 33; // Servo 1: CẤP PHÔI (GPIO 33)
const int PIN_SERVO_2 = 32; // Servo 2: GẠT PHÔI (GPIO 32)

// =============================================================================
// 2. THÔNG SỐ XUNG PWM & HÀNH TRÌNH TỪNG SERVO (50Hz = 20ms/xung)
// =============================================================================
const int SERVO_STOP_US = 1500; // Dừng hoàn toàn
const int SERVO_FWD_US  = 1200; // Quay thuận (Xung mạnh mẽ, đủ lực kéo cơ cấu đẩy/gạt)
const int SERVO_REV_US  = 1800; // Quay nghịch (Xung mạnh mẽ, đủ lực kéo cơ cấu đẩy/gạt)

// Cấu hình số xung riêng cho từng Servo:
// - Servo 1 (Cấp phôi): Thuận 65 xung, Nghịch 68 xung
const int SERVO1_FWD_PULSES = 65; // 65 * 20ms = 1300ms
const int SERVO1_REV_PULSES = 68; // 68 * 20ms = 1360ms

// - Servo 2 (Gạt phôi): Thuận 68 xung, Nghịch 70 xung
const int SERVO2_FWD_PULSES = 68; // 68 * 20ms = 1360ms
const int SERVO2_REV_PULSES = 70; // 70 * 20ms = 1400ms

const unsigned long CAMERA_STOP_TIME_MS = 8000; // Dừng 8 giây tại Camera

// Tốc độ băng tải
const unsigned long RAMP_UP_TIME_MS = 400;   
const float START_SPEED_SPS         = 300.0; 
const float MAX_SPEED_SPS           = 1000.0;

Servo servo1;
Servo servo2;

// =============================================================================
// 3. MÁY TRẠNG THÁI HỆ THỐNG
// =============================================================================
enum SystemState {
  STATE_IDLE,                 // Dừng nghỉ hoàn toàn
  STATE_SERVO1_FEED,          // Servo 1 cấp phôi trước (băng tải DỪNG), chờ Servo 1 hoàn tất
  STATE_FEED_AND_START,       // Băng tải tăng tốc (Servo 1 đã xong)
  STATE_RUN_WAIT_SENSOR1,     // Đang chạy chờ Cảm biến 1 (Camera)
  STATE_CAMERA_STOP_8S,       // Dừng 8s tại Camera
  STATE_RUN_WAIT_SENSOR2,     // Chạy tiếp chờ Cảm biến 2
  STATE_SERVO_EJECTING        // Cảm biến 2 thấy -> 2 servo chạy 70 xung
};

SystemState currentState = STATE_IDLE;

unsigned long stateStartTimeMs = 0;
unsigned long lastStepUs = 0;
float currentSpeedSps = START_SPEED_SPS;

// Chống rung nút bấm
int lastStartVal = HIGH;
int startState   = HIGH;
unsigned long lastStartDebounce = 0;

int lastStopVal  = HIGH;
int stopState    = HIGH;
unsigned long lastStopDebounce  = 0;

const unsigned long DEBOUNCE_DELAY_MS = 40;

// Trạng thái cảm biến trước đó để bắt cạnh
int prevSensor1 = HIGH;
int prevSensor2 = HIGH;

// =============================================================================
// 4. BỘ ĐIỀU KHIỂN SERVO THỰC THI ĐẢO CHIỀU ĐỘC LẬP TỪNG SERVO
// =============================================================================
enum ServoRunStage {
  STAGE_IDLE,        // Đang dừng
  STAGE_RUN_FWD,     // Đang quay thuận
  STAGE_PAUSE_100MS, // Nghỉ đệm 100ms
  STAGE_RUN_REV      // Đang quay nghịch
};

struct ServoUnit {
  ServoRunStage stage = STAGE_IDLE;
  int pulseCount = 0;
  unsigned long lastPulseTimeMs = 0;
  unsigned long pauseStartTimeMs = 0;
  int targetFwdPulses = 65; // Số xung quay thuận riêng
  int targetRevPulses = 68; // Số xung quay nghịch riêng
};

// Khởi tạo thông số cho từng Servo
ServoUnit s1 = {STAGE_IDLE, 0, 0, 0, SERVO1_FWD_PULSES, SERVO1_REV_PULSES};
ServoUnit s2 = {STAGE_IDLE, 0, 0, 0, SERVO2_FWD_PULSES, SERVO2_REV_PULSES};

// Kích hoạt chạy: Quay thuận -> Dừng 100ms -> Quay nghịch -> Dừng
void triggerMotion(Servo &srv, ServoUnit &unit, const char* name) {
  unit.stage = STAGE_RUN_FWD;
  unit.pulseCount = 0;
  unit.lastPulseTimeMs = millis();
  srv.writeMicroseconds(SERVO_FWD_US);
  Serial.printf("\n[%s] Bắt đầu quay thuận %d xung...\n", name, unit.targetFwdPulses);
}

// Cập nhật từng xung 20ms trong loop
void updateUnit(Servo &srv, ServoUnit &unit, const char* name) {
  unsigned long now = millis();

  switch (unit.stage) {
    case STAGE_RUN_FWD: {
      if (now - unit.lastPulseTimeMs >= 20) {
        unit.lastPulseTimeMs = now;
        unit.pulseCount++;

        if (unit.pulseCount % 30 == 0 || unit.pulseCount == unit.targetFwdPulses) {
          Serial.printf("[%s] Quay thuận: %d/%d xung\n", name, unit.pulseCount, unit.targetFwdPulses);
        }

        // Đủ xung thuận -> Dừng đệm 100ms
        if (unit.pulseCount >= unit.targetFwdPulses) {
          srv.writeMicroseconds(SERVO_STOP_US);
          unit.stage = STAGE_PAUSE_100MS;
          unit.pauseStartTimeMs = now;
        }
      }
      break;
    }

    case STAGE_PAUSE_100MS: {
      if (now - unit.pauseStartTimeMs >= 100) {
        unit.stage = STAGE_RUN_REV;
        unit.pulseCount = 0;
        unit.lastPulseTimeMs = now;
        srv.writeMicroseconds(SERVO_REV_US);
        Serial.printf("[%s] Đổi chiều quay nghịch %d xung về gốc...\n", name, unit.targetRevPulses);
      }
      break;
    }

    case STAGE_RUN_REV: {
      if (now - unit.lastPulseTimeMs >= 20) {
        unit.lastPulseTimeMs = now;
        unit.pulseCount++;

        if (unit.pulseCount % 30 == 0 || unit.pulseCount == unit.targetRevPulses) {
          Serial.printf("[%s] Quay nghịch: %d/%d xung\n", name, unit.pulseCount, unit.targetRevPulses);
        }

        // Đủ xung nghịch -> DỪNG HOÀN TOÀN
        if (unit.pulseCount >= unit.targetRevPulses) {
          srv.writeMicroseconds(SERVO_STOP_US);
          unit.stage = STAGE_IDLE;
          unit.pulseCount = 0;
          Serial.printf("[%s] HOÀN TẤT CHU KỲ (Biến xung = 0)\n", name);
        }
      }
      break;
    }

    case STAGE_IDLE:
    default:
      break;
  }
}

void updateServos() {
  updateUnit(servo1, s1, "SERVO 1");
  updateUnit(servo2, s2, "SERVO 2");
}

// =============================================================================
// 5. ĐIỀU KHIỂN BĂNG TẢI (STEPPER MOTOR)
// =============================================================================
float calculateRampSpeed(unsigned long elapsedTimeMs) {
  if (elapsedTimeMs >= RAMP_UP_TIME_MS) return MAX_SPEED_SPS;
  return START_SPEED_SPS + ((MAX_SPEED_SPS - START_SPEED_SPS) * (float)elapsedTimeMs / (float)RAMP_UP_TIME_MS);
}

void triggerStep(float sps) {
  if (sps <= 0.0) return;
  unsigned long intervalUs = (unsigned long)(1000000.0 / sps);
  unsigned long nowUs = micros();

  if (nowUs - lastStepUs >= intervalUs) {
    lastStepUs = nowUs;
    digitalWrite(PIN_STEP, HIGH);
    delayMicroseconds(5);
    digitalWrite(PIN_STEP, LOW);
  }
}

// =============================================================================
// 6. XỬ LÝ LỆNH DỪNG (STOP) VÀ KHỞI ĐỘNG (START)
// =============================================================================
void doStop() {
  Serial.println("\n=======================================================");
  Serial.println(">>> [LỆNH STOP]: DỪNG BĂNG TẢI & 2 SERVO, RESET XUNG VỀ 0");
  Serial.println("=======================================================\n");

  currentState = STATE_IDLE;
  currentSpeedSps = START_SPEED_SPS;

  // Dừng ngay lập tức 2 Servo
  servo1.writeMicroseconds(SERVO_STOP_US);
  servo2.writeMicroseconds(SERVO_STOP_US);

  s1.stage = STAGE_IDLE;
  s1.pulseCount = 0;

  s2.stage = STAGE_IDLE;
  s2.pulseCount = 0;
}

void doStart() {
  Serial.println("\n=======================================================");
  Serial.println(">>> [LỆNH START]: SERVO 1 CẤP PHÔI TRƯỚC, BĂNG TẢI CHỜ!");
  Serial.println("=======================================================\n");

  currentState = STATE_SERVO1_FEED;
  triggerMotion(servo1, s1, "SERVO 1 (CẤP PHÔI)");
}

// Xử lý nút bấm với bộ lọc chống rung
void checkButtons() {
  unsigned long now = millis();

  // Nhận lệnh phím từ Serial: 's' hoặc '1' = START | '0' hoặc 'x' = STOP
  if (Serial.available() > 0) {
    char c = Serial.read();
    if (c == '0' || c == 'x' || c == 'X' || c == 'e' || c == 'E') {
      doStop();
    } else if (c == 's' || c == 'S' || c == '1') {
      if (currentState == STATE_IDLE) {
        doStart();
      } else {
        Serial.println(">>> Đang chạy -> Cấp thêm 1 phôi mới (Servo 1)!");
        triggerMotion(servo1, s1, "SERVO 1 (CẤP PHÔI)");
      }
    }
  }

  // Nút START (GPIO 25 - Nối GND -> Ấn = LOW)
  int rStart = digitalRead(PIN_BTN_START);
  if (rStart != lastStartVal) lastStartDebounce = now;
  if ((now - lastStartDebounce) > DEBOUNCE_DELAY_MS) {
    if (rStart != startState) {
      startState = rStart;
      if (startState == LOW) { // Nhấn START
        if (currentState == STATE_IDLE) {
          doStart();
        } else {
          Serial.println(">>> [NÚT START] Cấp thêm 1 phôi mới!");
          triggerMotion(servo1, s1, "SERVO 1 (CẤP PHÔI)");
        }
      }
    }
  }
  lastStartVal = rStart;

  // Nút STOP (GPIO 26) - Bắt mọi trường hợp thay đổi (Ấn = DỪNG)
  int rStop = digitalRead(PIN_BTN_STOP);
  if (rStop != lastStopVal) lastStopDebounce = now;
  if ((now - lastStopDebounce) > DEBOUNCE_DELAY_MS) {
    if (rStop != stopState) {
      stopState = rStop;
      // Nhấn nút STOP:
      // - Nếu dùng nút NC (thường đóng nối GND): Bình thường LOW, khi ấn = HIGH
      // - Nếu dùng nút NO (thường mở nối GND): Bình thường HIGH, khi ấn = LOW
      // Để luôn luôn dừng được:
      if (stopState == HIGH) { 
        doStop();
      }
    }
  }
  lastStopVal = rStop;
}

// =============================================================================
// 7. SETUP
// =============================================================================
void setup() {
  Serial.begin(115200);
  delay(500);

  Serial.println("\n=======================================================");
  Serial.println("  HỆ THỐNG ĐIỀU KHIỂN BĂNG TẢI PCB + 2 SERVO SG90 360 ĐỘ");
  Serial.println("=======================================================");

  // 1. Chân Động cơ bước TB6600
  pinMode(PIN_STEP, OUTPUT);
  pinMode(PIN_DIR,  OUTPUT);
  pinMode(PIN_ENA,  OUTPUT);

  digitalWrite(PIN_DIR, DIR_FORWARD);
  digitalWrite(PIN_ENA, LOW); // Bật driver

  // 2. Chân Nút nhấn & Cảm biến
  pinMode(PIN_BTN_START, INPUT_PULLUP);
  pinMode(PIN_BTN_STOP,  INPUT_PULLUP);
  pinMode(PIN_SENSOR_1,  INPUT_PULLUP);
  pinMode(PIN_SENSOR_2,  INPUT_PULLUP);

  lastStartVal = digitalRead(PIN_BTN_START);
  startState   = lastStartVal;
  lastStopVal  = digitalRead(PIN_BTN_STOP);
  stopState    = lastStopVal;

  prevSensor1  = digitalRead(PIN_SENSOR_1);
  prevSensor2  = digitalRead(PIN_SENSOR_2);

  // 3. Khởi tạo 2 Servo
  ESP32PWM::allocateTimer(0);
  ESP32PWM::allocateTimer(1);
  ESP32PWM::allocateTimer(2);
  ESP32PWM::allocateTimer(3);

  servo1.setPeriodHertz(50);
  servo1.attach(PIN_SERVO_1, 1000, 2000);
  servo1.writeMicroseconds(SERVO_STOP_US); // 1500us

  servo2.setPeriodHertz(50);
  servo2.attach(PIN_SERVO_2, 1000, 2000);
  servo2.writeMicroseconds(SERVO_STOP_US); // 1500us

  Serial.println(">>> Sẵn sàng! Ấn nút START (GPIO 25) để bắt đầu...");
}

// =============================================================================
// 8. LOOP CHÍNH
// =============================================================================
void loop() {
  unsigned long now = millis();

  // 1. Quét nút nhấn & lệnh Serial
  checkButtons();

  // 2. Cập nhật xung phát cho 2 Servo (50Hz = đúng 20ms/xung)
  updateServos();

  // 3. Máy trạng thái điều khiển Băng tải & Cảm biến
  switch (currentState) {
    case STATE_IDLE:
      // Băng tải dừng
      break;

    // TRẠNG THÁI 0: SERVO 1 CẤP PHÔI - BĂNG TẢI ĐỨNG YÊN CHỜ
    case STATE_SERVO1_FEED: {
      // Không gọi triggerStep() -> băng tải đứng yên hoàn toàn
      // Khi Servo 1 hoàn tất (65 xung thuận + 68 xung nghịch) -> sang tăng tốc
      if (s1.stage == STAGE_IDLE) {
        Serial.println(">>> [SERVO 1 ĐÃ XONG] Bắt đầu khởi động băng tải...");
        currentState = STATE_FEED_AND_START;
        stateStartTimeMs = millis();
        currentSpeedSps = START_SPEED_SPS;
      }
      break;
    }

    // TRẠNG THÁI 1: BĂNG TẢI TĂNG TỐC (SERVO 1 ĐÃ XONG)
    case STATE_FEED_AND_START: {
      unsigned long elapsed = now - stateStartTimeMs;
      currentSpeedSps = calculateRampSpeed(elapsed);
      triggerStep(currentSpeedSps);

      if (elapsed >= RAMP_UP_TIME_MS) {
        currentState = STATE_RUN_WAIT_SENSOR1;
        Serial.println("[BĂNG TẢI] Đã đạt tốc độ tối đa -> Chờ phôi đến Cảm biến 1...");
      }
      break;
    }

    // TRẠNG THÁI 2: ĐANG CHẠY -> CHỜ CẢM BIẾN 1 (CAMERA)
    case STATE_RUN_WAIT_SENSOR1: {
      triggerStep(MAX_SPEED_SPS);

      int s1Val = digitalRead(PIN_SENSOR_1);
      // Chỉ kích hoạt 1 lần khi có phôi mới đi tới (bắt cạnh rơi)
      if (s1Val == SENSOR_DETECT_LEVEL && prevSensor1 != SENSOR_DETECT_LEVEL) {
        Serial.println("\n>>> [CẢM BIẾN 1] PHÁT HIỆN PHÔI TẠI CAMERA!");
        Serial.println(">>> [BĂNG TẢI] DỪNG ĐÚNG 8 GIÂY ĐỂ CHỤP ẢNH & KIỂM TRA...");
        
        stateStartTimeMs = now;
        currentState = STATE_CAMERA_STOP_8S;
      }
      prevSensor1 = s1Val;
      break;
    }

    // TRẠNG THÁI 3: DỪNG 8 GIÂY TẠI CAMERA
    case STATE_CAMERA_STOP_8S: {
      // Dừng băng tải hoàn toàn (không triggerStep)
      if (now - stateStartTimeMs >= CAMERA_STOP_TIME_MS) {
        Serial.println("\n>>> [CAMERA ĐÃ XONG] HẾT 8 GIÂY DỪNG!");
        Serial.println(">>> [BĂNG TẢI] TIẾP TỤC CHẠY ĐẾN KHI CẢM BIẾN 2 CÓ TÍN HIỆU...");
        
        stateStartTimeMs = now;
        currentState = STATE_RUN_WAIT_SENSOR2;
      }
      break;
    }

    // TRẠNG THÁI 4: CHẠY TIẾP -> CHỜ CẢM BIẾN 2 (CUỐI BĂNG TẢI)
    case STATE_RUN_WAIT_SENSOR2: {
      triggerStep(MAX_SPEED_SPS);

      int s2Val = digitalRead(PIN_SENSOR_2);
      // Chỉ kích hoạt 1 lần duy nhất khi phát hiện phôi mới (bắt cạnh rơi)
      if (s2Val == SENSOR_DETECT_LEVEL && prevSensor2 != SENSOR_DETECT_LEVEL) {
        Serial.println("\n>>> [CẢM BIẾN 2] PHÁT HIỆN PHÔI Ở CUỐI BĂNG TẢI!");
        Serial.println(">>> [BĂNG TẢI] DỪNG LẠI ĐỂ 2 SERVO HOẠT ĐỘNG!");
        Serial.println(">>> [CẢ 2 SERVO] KÍCH HOẠT QUAY 70 XUNG THUẬN -> NGHỊCH!");

        // Kích hoạt CẢ 2 SERVO: Servo 1 (65/68), Servo 2 (68/70)
        triggerMotion(servo1, s1, "SERVO 1");
        triggerMotion(servo2, s2, "SERVO 2");

        currentState = STATE_SERVO_EJECTING;
      }
      prevSensor2 = s2Val;
      break;
    }

    // TRẠNG THÁI 5: BĂNG TẢI DỪNG TRONG KHI 2 SERVO HOẠT ĐỘNG
    case STATE_SERVO_EJECTING: {
      // BĂNG TẢI DỪNG HOÀN TOÀN (KHÔNG gọi triggerStep) để gạt phôi an toàn, không bị xô lệch phôi!

      // Khi CẢ 2 SERVO đã hoàn thành đủ hành trình và về vị trí ban đầu (STAGE_IDLE):
      if (s1.stage == STAGE_IDLE && s2.stage == STAGE_IDLE) {
        Serial.println("\n>>> [HOÀN TẤT GẠT PHÔI] Cả 2 Servo đã thu về vị trí ban đầu (Biến xung = 0).");
        Serial.println(">>> [BĂNG TẢI] TIẾP TỤC CHẠY ĐÓN PHÔI MỚI TẠI CẢM BIẾN 1.\n");

        currentState = STATE_RUN_WAIT_SENSOR1;
      }
      break;
    }

    default:
      currentState = STATE_IDLE;
      break;
  }
}

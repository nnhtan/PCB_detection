#include <Arduino.h>
#include <ESP32Servo.h>

// =============================================================================
// 1. CẤU HÌNH CHÂN GPIO & ĐẶC TÍNH NÚT NHẤN
// =============================================================================
// Động cơ bước (Driver TB6600 / DM542)
const int PIN_STEP = 5;  // Chân xung PUL+ (GPIO 5)
const int PIN_DIR  = 19; // Chân chiều DIR+ (GPIO 19)
const int PIN_ENA  = 18; // Chân ENA+ (LOW = Bật motor, HIGH = Ngắt lực)

// Chiều quay động cơ băng tải (LOW <-> HIGH)
const int DIR_FORWARD = LOW; 

// Nút nhấn START (GPIO 25: Thường mở NO) & STOP (GPIO 26: Thường đóng NC)
const int PIN_BTN_START = 25; // Nút START: Thường mở (NO)
const int PIN_BTN_STOP  = 26; // Nút STOP : Thường đóng (NC)

// MỨC TÍCH CỰC KHI ẤN NÚT:
// - Nút START thường mở (NO) nối GND: Bình thường = HIGH, Khi ẤN = LOW
const int BTN_START_PRESSED_LEVEL = LOW;

// - Nút STOP thường đóng (NC) nối GND: Bình thường = LOW, Khi ẤN = HIGH
const int BTN_STOP_PRESSED_LEVEL  = HIGH;

// 2 CẢM BIẾN TIỆM CẬN (Mức LOW khi phát hiện phôi - loại NPN)
const int PIN_SENSOR_1 = 16; // Cảm biến 1: Vị trí CAMERA (GPIO 16)
const int PIN_SENSOR_2 = 17; // Cảm biến 2: Vị trí CUỐI BĂNG TẢI (GPIO 17)
const int SENSOR_DETECT_LEVEL = LOW; // Mức tích cực khi thấy phôi

// 2 SERVO SG90
const int PIN_SERVO_1 = 33; // Servo 1: CẤP PHÔI (GPIO 33)
const int PIN_SERVO_2 = 32; // Servo 2: GẠT PHÔI CUỐI BĂNG TẢI (GPIO 32)

// =============================================================================
// 2. THÔNG SỐ SERVO & THỜI GIAN
// =============================================================================
const int SERVO_IDLE_ANGLE    = 90; // 90: DỪNG HOÀN TOÀN (vị trí nghỉ)
const int SERVO1_ACTION_ANGLE = 0;  // 0 : Góc quay Servo 1 cấp phôi
const int SERVO2_ACTION_ANGLE = 0;  // 0 : Góc quay Servo 2 gạt phôi (hoặc 180 tùy hướng)

const unsigned long SERVO1_FEED_TIME_MS  = 600; // Thời gian Servo 1 đẩy phôi (ms)
const unsigned long SERVO2_EJECT_TIME_MS = 800; // Thời gian Servo 2 gạt phôi (ms)

// Thời gian dừng băng tải tại vị trí Cảm biến 1 (Camera): 8 GIÂY (8000ms)
const unsigned long CAMERA_STOP_TIME_MS = 8000;

// =============================================================================
// 3. THÔNG SỐ VẬN HÀNH ĐỘNG CƠ BƯỚC (BĂNG TẢI)
// =============================================================================
const unsigned long RAMP_UP_TIME_MS = 400;   // Thời gian tăng tốc (ms)
const float START_SPEED_SPS         = 300.0; // Tốc độ khởi động (xung/giây)
const float MAX_SPEED_SPS           = 1000.0;// Tốc độ chạy chính (xung/giây)

// Đối tượng Servo
Servo servo1;
Servo servo2;

// =============================================================================
// 4. MÁY TRẠNG THÁI (FINITE STATE MACHINE)
// =============================================================================
enum SystemState {
  STATE_IDLE,                 // Băng tải dừng, chờ ấn nút START
  STATE_FEED_AND_START,       // Băng tải chạy tăng tốc + Servo 1 cấp phôi
  STATE_RUN_WAIT_SENSOR1,     // Băng tải đang chạy, chờ phôi tới Cảm biến 1
  STATE_CAMERA_STOP_8S,       // Cảm biến 1 thấy phôi -> DỪNG 8 GIÂY
  STATE_RUN_WAIT_SENSOR2,     // Sau 8s: Băng tải tiếp tục chạy, chờ Cảm biến 2
  STATE_SERVO2_EJECTING       // Cảm biến 2 thấy phôi -> Servo 2 gạt, BĂNG TẢI VẪN TIẾP TỤC CHẠY
};

SystemState currentState = STATE_IDLE;

// Biến quản lý thời gian và tốc độ
unsigned long stateStartTimeMs = 0;
unsigned long servo1StartTimeMs = 0;
unsigned long servo2StartTimeMs = 0;
unsigned long lastStepUs = 0;
float currentSpeedSps = START_SPEED_SPS;

// Biến lưu trạng thái nút bấm trước đó để bắt cạnh nhấn
int prevStartBtnVal = LOW;
int prevStopBtnVal  = HIGH;

// Biến theo dõi trạng thái cảm biến để in log debug
int prevSensor1State = -1;
int prevSensor2State = -1;

// Cờ trạng thái servo
bool isServo1Running = false;
bool isServo2Running = false;

// =============================================================================
// 5. HÀM PHÁT XUNG TB6600 (NON-BLOCKING)
// =============================================================================
void triggerStep(float speedSps) {
  if (speedSps <= 0) return;
  unsigned long intervalUs = (unsigned long)(1000000.0 / speedSps);
  unsigned long nowUs = micros();

  if (nowUs - lastStepUs >= intervalUs) {
    lastStepUs = nowUs;
    digitalWrite(PIN_STEP, HIGH);
    delayMicroseconds(10); // Xung 10us chuẩn TB6600
    digitalWrite(PIN_STEP, LOW);
  }
}

// Dừng toàn bộ hoạt động khẩn cấp
void stopAll() {
  Serial.println("\n=======================================================");
  Serial.println("[STOP] ĐÃ DỪNG TOÀN BỘ HOẠT ĐỘNG!");
  Serial.println("  - Băng tải: DỪNG LẬP TỨC");
  Serial.println("  - Servo 1 & 2: VỀ VỊ TRÍ NGHỈ (90 độ)");
  Serial.println("  - Trạng thái: Chuyển về IDLE");
  Serial.println("=======================================================\n");

  servo1.write(SERVO_IDLE_ANGLE);
  servo2.write(SERVO_IDLE_ANGLE);
  isServo1Running = false;
  isServo2Running = false;

  currentState = STATE_IDLE;
  currentSpeedSps = START_SPEED_SPS;
}

// Kích hoạt Servo 1 cấp phôi
void triggerServo1Feed() {
  if (!isServo1Running) {
    Serial.println("\n[SERVO 1] Kích hoạt Servo 1 đẩy phôi vào băng tải!");
    servo1.write(SERVO1_ACTION_ANGLE);
    isServo1Running = true;
    servo1StartTimeMs = millis();
  }
}

// Bắt đầu chu trình khi ấn nút START
void startCycle() {
  Serial.println("\n=======================================================");
  Serial.println("[START] BẮT ĐẦU CHU TRÌNH:");
  Serial.println("  1. Băng tải bắt đầu chạy");
  Serial.println("  2. Servo 1 đẩy phôi vào băng tải");
  Serial.println("=======================================================");

  triggerServo1Feed();

  // Khởi động chạy băng tải
  currentState = STATE_FEED_AND_START;
  stateStartTimeMs = millis();
  currentSpeedSps = START_SPEED_SPS;
}

// =============================================================================
// 6. KIỂM TRA NÚT BẤM (START Thường đóng NC / STOP Thường mở NO) & SERIAL
// =============================================================================
void checkButtonsAndCommands() {
  // 1. Lệnh điều khiển qua Serial Monitor: 's' = Start, 'x' = Stop
  if (Serial.available() > 0) {
    char cmd = Serial.read();
    if (cmd == 's' || cmd == 'S') {
      if (currentState == STATE_IDLE) {
        Serial.println("[SERIAL] Nhận lệnh START!");
        startCycle();
      } else {
        Serial.println("[SERIAL] Đang chạy -> Kích hoạt Servo 1 cấp thêm phôi!");
        triggerServo1Feed();
      }
    } else if (cmd == 'x' || cmd == 'X' || cmd == 'e' || cmd == 'E') {
      Serial.println("[SERIAL] Nhận lệnh STOP!");
      stopAll();
    }
  }

  // 2. Nút STOP (Thường đóng NC nối GND -> Bình thường = LOW, Khi BẤM = HIGH)
  int stopVal = digitalRead(PIN_BTN_STOP);
  if (stopVal == BTN_STOP_PRESSED_LEVEL && prevStopBtnVal != BTN_STOP_PRESSED_LEVEL) {
    Serial.println("\n>>> [NÚT NHẤN] Nút STOP (Thường đóng - GPIO 26) ĐÃ ĐƯỢC ẤN!");
    stopAll();
  }
  prevStopBtnVal = stopVal;

  // 3. Nút START (Thường mở NO nối GND -> Bình thường = HIGH, Khi BẤM = LOW)
  int startVal = digitalRead(PIN_BTN_START);
  if (startVal == BTN_START_PRESSED_LEVEL && prevStartBtnVal != BTN_START_PRESSED_LEVEL) {
    if (currentState == STATE_IDLE) {
      Serial.println("\n>>> [NÚT NHẤN] Nút START (Thường mở - GPIO 25) ĐÃ ĐƯỢC ẤN!");
      startCycle();
    } else {
      Serial.println("\n>>> [NÚT NHẤN] Băng tải đang chạy -> Kích hoạt Servo 1 cấp phôi tiếp theo!");
      triggerServo1Feed();
    }
  }
  prevStartBtnVal = startVal;
}

// =============================================================================
// 7. THEO DÕI TRẠNG THÁI CẢM BIẾN (DEBUG SERIAL)
// =============================================================================
void monitorSensors() {
  int s1 = digitalRead(PIN_SENSOR_1);
  if (s1 != prevSensor1State) {
    prevSensor1State = s1;
    if (s1 == SENSOR_DETECT_LEVEL) {
      Serial.println("[DEBUG SENSOR 1 - GPIO 16] -> CÓ PHÔI");
    } else {
      Serial.println("[DEBUG SENSOR 1 - GPIO 16] -> Không có phôi");
    }
  }

  int s2 = digitalRead(PIN_SENSOR_2);
  if (s2 != prevSensor2State) {
    prevSensor2State = s2;
    if (s2 == SENSOR_DETECT_LEVEL) {
      Serial.println("[DEBUG SENSOR 2 - GPIO 17] -> CÓ PHÔI");
    } else {
      Serial.println("[DEBUG SENSOR 2 - GPIO 17] -> Không có phôi");
    }
  }
}

// =============================================================================
// 8. CẬP NHẬT TRẠNG THÁI SERVO (NON-BLOCKING)
// =============================================================================
void updateServos() {
  unsigned long now = millis();

  // Servo 1: Cấp phôi xong -> Thu về 90 độ
  if (isServo1Running && (now - servo1StartTimeMs >= SERVO1_FEED_TIME_MS)) {
    servo1.write(SERVO_IDLE_ANGLE);
    isServo1Running = false;
    Serial.println("[SERVO 1] Đã đẩy phôi xong -> Thu về vị trí nghỉ (90 độ).");
  }

  // Servo 2: Gạt phôi xong -> Thu về 90 độ
  if (isServo2Running && (now - servo2StartTimeMs >= SERVO2_EJECT_TIME_MS)) {
    servo2.write(SERVO_IDLE_ANGLE);
    isServo2Running = false;
    Serial.println("[SERVO 2] Đã gạt phôi xong -> Thu về vị trí nghỉ (90 độ).");
  }
}

// =============================================================================
// 9. SETUP
// =============================================================================
void setup() {
  Serial.begin(115200);
  delay(500);

  Serial.println("\n=======================================================");
  Serial.println("  HỆ THỐNG ĐIỀU KHIỂN BĂNG TẢI PCB + 2 SERVO + 2 SENSOR");
  Serial.println("=======================================================");
  Serial.println("  - Nút START (GPIO 25): Thường mở   (NO -> Ấn = LOW)");
  Serial.println("  - Nút STOP  (GPIO 26): Thường đóng (NC -> Ấn = HIGH)");
  Serial.println("  - Cảm biến 1 (Camera): GPIO 16");
  Serial.println("  - Cảm biến 2 (Cuối)  : GPIO 17");
  Serial.println("  - Servo 1 (Cấp phôi) : GPIO 33");
  Serial.println("  - Servo 2 (Gạt phôi) : GPIO 32");
  Serial.println("=======================================================\n");

  // 1. Cấu hình chân Động cơ bước TB6600
  pinMode(PIN_STEP, OUTPUT);
  pinMode(PIN_DIR,  OUTPUT);
  pinMode(PIN_ENA,  OUTPUT);

  digitalWrite(PIN_DIR, DIR_FORWARD); // Chiều quay băng tải
  digitalWrite(PIN_ENA, LOW);         // Enable driver

  // 2. Cấu hình nút nhấn & cảm biến
  pinMode(PIN_BTN_START, INPUT_PULLUP);
  pinMode(PIN_BTN_STOP,  INPUT_PULLUP);
  pinMode(PIN_SENSOR_1,  INPUT_PULLUP);
  pinMode(PIN_SENSOR_2,  INPUT_PULLUP);

  prevStartBtnVal = digitalRead(PIN_BTN_START);
  prevStopBtnVal  = digitalRead(PIN_BTN_STOP);
  prevSensor1State = digitalRead(PIN_SENSOR_1);
  prevSensor2State = digitalRead(PIN_SENSOR_2);

  Serial.printf("[KHỞI TẠO] Trạng thái chân Nút START (GPIO 26): %s\n", (prevStartBtnVal == LOW ? "LOW (Đang đóng tiếp điểm - OK)" : "HIGH (Đang hở tiếp điểm)"));
  Serial.printf("[KHỞI TẠO] Trạng thái chân Nút STOP  (GPIO 27): %s\n", (prevStopBtnVal == HIGH ? "HIGH (Đang hở tiếp điểm - OK)" : "LOW (Đang đóng tiếp điểm)"));

  // 3. Cấu hình 2 Servo SG90
  ESP32PWM::allocateTimer(0);
  ESP32PWM::allocateTimer(1);
  ESP32PWM::allocateTimer(2);
  ESP32PWM::allocateTimer(3);

  servo1.setPeriodHertz(50);
  servo1.write(SERVO_IDLE_ANGLE);
  servo1.attach(PIN_SERVO_1, 1000, 2000);

  servo2.setPeriodHertz(50);
  servo2.write(SERVO_IDLE_ANGLE);
  servo2.attach(PIN_SERVO_2, 1000, 2000);

  delay(500);

  currentState = STATE_IDLE;
  Serial.println("[HỆ THỐNG] SẴN SÀNG! Vui lòng ấn nút START (GPIO 26) để bắt đầu chu trình.");
}

// =============================================================================
// 10. LOOP CHÍNH
// =============================================================================
void loop() {
  // 1. Lắng nghe nút nhấn START/STOP và lệnh Serial
  checkButtonsAndCommands();

  // 2. In log debug nếu cảm biến thay đổi trạng thái
  monitorSensors();

  // 3. Cập nhật vị trí Servo non-blocking
  updateServos();

  unsigned long now = millis();

  switch (currentState) {

    // -------------------------------------------------------------------------
    // TRẠNG THÁI 0: CHỜ LỆNH START (Băng tải đứng yên)
    // -------------------------------------------------------------------------
    case STATE_IDLE:
      // Băng tải dừng hoàn toàn
      break;

    // -------------------------------------------------------------------------
    // TRẠNG THÁI 1: BĂNG TẢI KHỞI ĐỘNG CHẠY + SERVO 1 CẤP PHÔI
    // -------------------------------------------------------------------------
    case STATE_FEED_AND_START: {
      unsigned long elapsed = now - stateStartTimeMs;
      if (elapsed < RAMP_UP_TIME_MS) {
        float progress = (float)elapsed / (float)RAMP_UP_TIME_MS;
        currentSpeedSps = START_SPEED_SPS + progress * (MAX_SPEED_SPS - START_SPEED_SPS);
        triggerStep(currentSpeedSps);
      } else {
        currentSpeedSps = MAX_SPEED_SPS;
        currentState = STATE_RUN_WAIT_SENSOR1;
        Serial.println("[BĂNG TẢI] Đang chạy tốc độ MAX, đưa phôi tới Cảm biến 1...");
      }
      break;
    }

    // -------------------------------------------------------------------------
    // TRẠNG THÁI 2: BĂNG TẢI ĐANG CHẠY -> CHỜ PHÔI TỚI CẢM BIẾN 1 (CAMERA)
    // -------------------------------------------------------------------------
    case STATE_RUN_WAIT_SENSOR1: {
      // Duy trì băng tải chạy
      triggerStep(MAX_SPEED_SPS);

      // Khi phôi đến vị trí CẢM BIẾN 1 (GPIO 16):
      if (digitalRead(PIN_SENSOR_1) == SENSOR_DETECT_LEVEL) {
        Serial.println("\n>>> [CẢM BIẾN 1] PHÁT HIỆN PHÔI!");
        Serial.println(">>> [BĂNG TẢI] DỪNG 8 GIÂY ĐỂ CAMERA QUÉT & XỬ LÝ ẢNH...");
        
        currentState = STATE_CAMERA_STOP_8S;
        stateStartTimeMs = now; // Bắt đầu tính giờ 8s dừng
      }
      break;
    }

    // -------------------------------------------------------------------------
    // TRẠNG THÁI 3: DỪNG BĂNG TẢI 8 GIÂY TẠI CẢM BIẾN 1
    // -------------------------------------------------------------------------
    case STATE_CAMERA_STOP_8S: {
      // KHÔNG gọi triggerStep() -> Băng tải dừng hoàn toàn
      if (now - stateStartTimeMs >= CAMERA_STOP_TIME_MS) {
        Serial.println("\n>>> [CAMERA] ĐÃ HẾT 8 GIÂY DỪNG!");
        Serial.println(">>> [BĂNG TẢI] TIẾP TỤC CHẠY ĐẾN KHI CẢM BIẾN 2 CÓ TÍN HIỆU...");
        
        stateStartTimeMs = now;
        currentState = STATE_RUN_WAIT_SENSOR2;
      }
      break;
    }

    // -------------------------------------------------------------------------
    // TRẠNG THÁI 4: BĂNG TẢI TIẾP TỤC CHẠY -> CHỜ CẢM BIẾN 2 (CUỐI BĂNG TẢI)
    // -------------------------------------------------------------------------
    case STATE_RUN_WAIT_SENSOR2: {
      // Băng tải tiếp tục chạy
      triggerStep(MAX_SPEED_SPS);

      // Khi phôi đến vị trí CẢM BIẾN 2 (GPIO 17):
      if (digitalRead(PIN_SENSOR_2) == SENSOR_DETECT_LEVEL) {
        Serial.println("\n>>> [CẢM BIẾN 2] PHÁT HIỆN PHÔI Ở CUỐI BĂNG TẢI!");
        Serial.println(">>> [CẢ 2 SERVO] KÍCH HOẠT CẢ SERVO 1 VÀ SERVO 2 HOẠT ĐỘNG! (BĂNG TẢI TIẾP TỤC CHẠY)");

        // Kích hoạt Servo 1 hoạt động
        servo1.write(SERVO1_ACTION_ANGLE);
        isServo1Running = true;
        servo1StartTimeMs = now;

        // Kích hoạt Servo 2 hoạt động
        servo2.write(SERVO2_ACTION_ANGLE);
        isServo2Running = true;
        servo2StartTimeMs = now;

        currentState = STATE_SERVO2_EJECTING;
      }
      break;
    }

    // -------------------------------------------------------------------------
    // TRẠNG THÁI 5: CẢ 2 SERVO HOẠT ĐỘNG VÀ BĂNG TẢI VẪN TIẾP TỤC CHẠY
    // -------------------------------------------------------------------------
    case STATE_SERVO2_EJECTING: {
      // BĂNG TẢI VẪN TIẾP TỤC CHẠY!
      triggerStep(MAX_SPEED_SPS);

      // Khi cả 2 Servo đã hoàn thành gạt phôi và thu về 90 độ:
      if (!isServo1Running && !isServo2Running) {
        Serial.println("\n>>> [HOÀN TẤT GẠT PHÔI] Cả 2 Servo đã thu về vị trí nghỉ. BĂNG TẢI TIẾP TỤC CHẠY.");
        Serial.println(">>> Sẵn sàng nhận phôi tiếp theo.\n");

        // Băng tải vẫn chạy liên tục, chuyển về trạng thái sẵn sàng đón phôi mới tại Cảm biến 1
        currentState = STATE_RUN_WAIT_SENSOR1;
      }
      break;
    }

    default:
      currentState = STATE_IDLE;
      break;
  }
}
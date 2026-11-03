"""Serial communication with the ESP32 conveyor controller."""

import threading
import time
from queue import Empty, Queue

import serial


class ConveyorCommunicationError(RuntimeError):
    """Raised when the conveyor controller cannot accept a result."""


def normalize_verdict(is_ok: bool) -> str:
    if not isinstance(is_ok, bool):
        raise ValueError("Kết quả gửi ESP phải là true (đạt) hoặc false (không đạt).")
    return "OK" if is_ok else "FAULTY"


class ConveyorClient:
    def __init__(
        self,
        port: str = "COM5",
        baudrate: int = 115200,
        timeout: float = 5.0,
    ) -> None:
        if not port.strip():
            raise ValueError("Cổng Serial không được để trống.")
        if baudrate <= 0:
            raise ValueError("Baud rate phải lớn hơn 0.")
        if timeout <= 0:
            raise ValueError("Timeout phải lớn hơn 0.")

        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self._connection: serial.Serial | None = None
        self._lock = threading.Lock()
        self._reader_lock = threading.Lock()
        self._reader_stop = threading.Event()
        self._reader_thread: threading.Thread | None = None
        self._reader_error: Exception | None = None
        self._responses: Queue[str] = Queue()
        self._events: Queue[str] = Queue()

    def connect(self) -> None:
        if self._connection is not None and self._connection.is_open:
            return

        connection: serial.Serial | None = None
        try:
            connection = serial.Serial(
                port=None,
                baudrate=self.baudrate,
                timeout=0.2,
                write_timeout=self.timeout,
            )
            connection.dtr = False
            connection.rts = False
            connection.port = self.port
            connection.open()
            connection.reset_input_buffer()
        except (serial.SerialException, OSError) as exc:
            if connection is not None and connection.is_open:
                connection.close()
            raise ConveyorCommunicationError(
                f"Không thể mở cổng ESP {self.port} ở {self.baudrate} baud: {exc}"
            ) from exc

        self._connection = connection
        self._reader_error = None
        self._start_reader(connection)

    def _start_reader(self, connection: serial.Serial) -> None:
        with self._reader_lock:
            if self._reader_thread is not None and self._reader_thread.is_alive():
                return
            self._reader_stop.clear()
            self._reader_thread = threading.Thread(
                target=self._read_serial,
                args=(connection,),
                name="conveyor-serial-reader",
                daemon=True,
            )
            self._reader_thread.start()

    def _read_serial(self, connection: serial.Serial) -> None:
        while not self._reader_stop.is_set():
            try:
                response = connection.readline().decode(
                    "ascii", errors="replace"
                ).strip()
            except (serial.SerialException, OSError) as exc:
                if not self._reader_stop.is_set():
                    self._reader_error = exc
                return

            if response.startswith("EVENT:"):
                self._events.put(response.removeprefix("EVENT:"))
            elif response.startswith(("ACK:", "NACK:")):
                self._responses.put(response)

    def wait_for_event(self, timeout: float = 1.0) -> str | None:
        if timeout < 0:
            raise ValueError("Timeout chờ sự kiện không được âm.")

        connection = self._connection
        if connection is None or not connection.is_open:
            raise ConveyorCommunicationError(
                "Kết nối ESP chưa sẵn sàng; hãy kiểm tra cấu hình Serial."
            )
        self._start_reader(connection)

        try:
            return self._events.get(timeout=timeout)
        except Empty:
            if self._reader_error is not None:
                raise ConveyorCommunicationError(
                    f"Lỗi nhận dữ liệu từ ESP qua {self.port}: {self._reader_error}"
                ) from self._reader_error
            return None

    def _send_command(
        self, command: str, expected_ack: str, action: str
    ) -> None:
        with self._lock:
            connection = self._connection
            if connection is None or not connection.is_open:
                raise ConveyorCommunicationError(
                    "Kết nối ESP chưa sẵn sàng; hãy kiểm tra cấu hình Serial."
                )

            try:
                self._start_reader(connection)
                connection.write(f"{command}\n".encode("ascii"))
                connection.flush()
                deadline = time.monotonic() + self.timeout
                while time.monotonic() < deadline:
                    remaining = deadline - time.monotonic()
                    try:
                        response = self._responses.get(
                            timeout=min(0.2, remaining)
                        )
                    except Empty:
                        if self._reader_error is not None:
                            raise ConveyorCommunicationError(
                                f"Lỗi nhận dữ liệu từ ESP qua {self.port}: "
                                f"{self._reader_error}"
                            ) from self._reader_error
                        continue
                    if response == expected_ack:
                        return
                    if response.startswith("NACK:"):
                        raise ConveyorCommunicationError(
                            f"ESP từ chối {action}: {response}"
                        )
            except ConveyorCommunicationError:
                raise
            except (serial.SerialException, OSError) as exc:
                raise ConveyorCommunicationError(
                    f"Lỗi truyền dữ liệu đến ESP qua {self.port}: {exc}"
                ) from exc

        raise ConveyorCommunicationError(
            f"ESP không xác nhận {action} trong {self.timeout:g} giây."
        )

    def send_result(self, is_ok: bool) -> str:
        verdict = normalize_verdict(is_ok)
        self._send_command(
            f"RESULT:{verdict}",
            f"ACK:RESULT:{verdict}",
            f"kết quả {verdict}",
        )
        return verdict

    def resume(self) -> str:
        self._send_command("RESUME", "ACK:RESUME", "lệnh tiếp tục băng tải")
        return "RESUMED"

    def close(self) -> None:
        self._reader_stop.set()
        connection = self._connection
        if connection is not None:
            connection.close()
            self._connection = None
        if self._reader_thread is not None:
            self._reader_thread.join(timeout=1.0)
            self._reader_thread = None

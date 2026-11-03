import asyncio
from queue import Empty, Queue
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import HTTPException

from backend.app.api.v1.common import (
    inspection_payload,
    resume_conveyor,
    send_result_to_conveyor,
)
from backend.app.api.v1.system import (
    ConveyorResult,
    resume_conveyor_belt,
    send_conveyor_result,
    wait_for_conveyor_event,
)
from backend.app.services.conveyor import (
    ConveyorClient,
    ConveyorCommunicationError,
    normalize_verdict,
)


class FakeSerial:
    def __init__(self, responses):
        self.responses = Queue()
        for response in responses:
            self.responses.put(response)
        self.is_open = True
        self.writes = []

    def write(self, data):
        self.writes.append(data)

    def flush(self):
        pass

    def readline(self):
        try:
            return self.responses.get(timeout=0.1)
        except Empty:
            return b""


class FakeConveyor:
    def __init__(self):
        self.results = []

    def send_result(self, result):
        self.results.append(result)


class ConveyorCommunicationTests(unittest.TestCase):
    def test_only_binary_inspection_verdicts_are_sent_to_conveyor(self):
        result = {
            "status": "UNALIGNED",
            "status_vi": "Chưa căn chỉnh",
            "aligned": False,
            "qa_available": False,
            "comparison_scope": "",
            "comparison_note": "",
            "alignment_method": "",
            "alignment_error": None,
            "alignment_transform": None,
            "orientation_transform": None,
            "crop_method": "",
            "stats": {"matched": 0, "missing": 0, "extra": 0},
            "inventory_comparison": {},
            "class_counts": {},
            "detections": [],
            "matched_components": [],
            "class_mismatch_details": [],
            "missing_components_details": [],
            "extra_components_details": [],
            "latency_ms": 0,
            "annotated_image": None,
            "board_crop_image": None,
        }

        self.assertIsNone(inspection_payload(result)["is_ok"])
        result["status"] = "OK"
        self.assertIs(inspection_payload(result)["is_ok"], True)
        result["status"] = "FAULTY"
        self.assertIs(inspection_payload(result)["is_ok"], False)

    def test_normalizes_only_binary_result(self):
        self.assertEqual(normalize_verdict(True), "OK")
        self.assertEqual(normalize_verdict(False), "FAULTY")

    def test_rejects_non_boolean_result(self):
        with self.assertRaises(ValueError):
            normalize_verdict("OK")

    def test_sends_result_and_waits_for_matching_ack(self):
        client = ConveyorClient()
        fake_serial = FakeSerial(
            [b"debug message\r\n", b"ACK:RESULT:OK\r\n"]
        )
        client._connection = fake_serial

        self.assertEqual(client.send_result(True), "OK")
        self.assertEqual(fake_serial.writes, [b"RESULT:OK\n"])

    def test_sends_not_ok_as_faulty(self):
        client = ConveyorClient()
        fake_serial = FakeSerial([b"ACK:RESULT:FAULTY\r\n"])
        client._connection = fake_serial

        self.assertEqual(client.send_result(False), "FAULTY")
        self.assertEqual(fake_serial.writes, [b"RESULT:FAULTY\n"])

    def test_sends_resume_command_and_waits_for_ack(self):
        client = ConveyorClient()
        fake_serial = FakeSerial([b"ACK:RESUME\r\n"])
        client._connection = fake_serial

        self.assertEqual(client.resume(), "RESUMED")
        self.assertEqual(fake_serial.writes, [b"RESUME\n"])

    def test_surfaces_esp_rejection(self):
        client = ConveyorClient()
        client._connection = FakeSerial([b"NACK:NOT_WAITING_FOR_RESULT\r\n"])

        with self.assertRaises(ConveyorCommunicationError):
            client.send_result(False)

    def test_receives_sensor_event_from_esp(self):
        client = ConveyorClient()
        client._connection = FakeSerial(
            [b"debug output\r\n", b"EVENT:SENSOR1\r\n"]
        )

        self.assertEqual(client.wait_for_event(timeout=1), "SENSOR1")

    def test_api_waits_for_conveyor_event(self):
        class FakeConveyor:
            def wait_for_event(self, timeout):
                self.timeout = timeout
                return "SENSOR1"

        conveyor = FakeConveyor()
        request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(
            conveyor=conveyor
        )))
        with patch(
            "backend.app.api.v1.system.config.CONVEYOR_ENABLED", True
        ):
            response = asyncio.run(
                wait_for_conveyor_event(request, timeout=0.5)
            )

        self.assertEqual(response, {"enabled": True, "event": "SENSOR1"})
        self.assertEqual(conveyor.timeout, 0.5)

    def test_api_disables_conveyor_event_polling_when_not_configured(self):
        request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(
            conveyor=None
        )))
        with patch(
            "backend.app.api.v1.system.config.CONVEYOR_ENABLED", False
        ):
            response = asyncio.run(wait_for_conveyor_event(request))

        self.assertEqual(response, {"enabled": False, "event": None})

    def test_api_sends_result_to_enabled_conveyor(self):
        conveyor = FakeConveyor()
        request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(
            conveyor=conveyor
        )))
        with patch(
            "backend.app.api.v1.common.config.CONVEYOR_ENABLED", True
        ):
            asyncio.run(
                send_result_to_conveyor(request, False)
            )

        self.assertEqual(conveyor.results, [False])

    def test_api_sends_resume_to_enabled_conveyor(self):
        class FakeConveyor:
            def __init__(self):
                self.resumed = False

            def resume(self):
                self.resumed = True

        conveyor = FakeConveyor()
        request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(
            conveyor=conveyor
        )))
        with patch(
            "backend.app.api.v1.common.config.CONVEYOR_ENABLED", True
        ):
            asyncio.run(resume_conveyor(request))

        self.assertTrue(conveyor.resumed)

    def test_api_receives_displayed_boolean_result(self):
        request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace()))
        with patch(
            "backend.app.api.v1.system.send_result_to_conveyor"
        ) as send_result:
            response = asyncio.run(
                send_conveyor_result(ConveyorResult(is_ok=True), request)
            )

        send_result.assert_awaited_once_with(request, True)
        self.assertEqual(response["sent"], "OK")

    def test_api_resumes_conveyor_after_scan(self):
        request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace()))
        with patch(
            "backend.app.api.v1.system.resume_conveyor"
        ) as resume:
            response = asyncio.run(resume_conveyor_belt(request))

        resume.assert_awaited_once_with(request)
        self.assertEqual(response, {"status": "success", "sent": "RESUME"})

    def test_api_surfaces_conveyor_failure(self):
        class FailedConveyor:
            def send_result(self, result):
                raise ConveyorCommunicationError("ESP did not respond")

        request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(
            conveyor=FailedConveyor()
        )))
        with patch(
            "backend.app.api.v1.common.config.CONVEYOR_ENABLED", True
        ):
            with self.assertRaises(HTTPException) as error:
                asyncio.run(
                    send_result_to_conveyor(request, True)
                )

        self.assertEqual(error.exception.status_code, 503)


if __name__ == "__main__":
    unittest.main()

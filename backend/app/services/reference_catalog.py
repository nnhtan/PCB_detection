import json
import os
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Any

import numpy as np


class ReferenceCatalog:
    def __init__(
        self,
        pipeline: Any,
        storage_dir: Path,
        operation_lock: RLock,
        legacy_detected_path: Path,
    ) -> None:
        self.pipeline = pipeline
        self.storage_dir = storage_dir
        self.catalog_path = storage_dir / "reference_catalog.json"
        self.references_dir = storage_dir / "references"
        self.operation_lock = operation_lock
        self.legacy_detected_path = legacy_detected_path
        self.default_reference_paths = (
            pipeline.ref_json_path,
            pipeline.ref_image_path,
        )
        self.references: list[dict[str, Any]] = []
        self.active_reference_id: str | None = None
        self._load_or_seed()

    def _reference_paths(self, reference_id: str) -> dict[str, Path]:
        folder = self.references_dir / reference_id
        return {
            "folder": folder,
            "image": folder / "reference.jpg",
            "components": folder / "reference_components.json",
            "annotated": folder / "reference_detected.jpg",
        }

    def _persist(self) -> None:
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        temporary_path = self.catalog_path.with_suffix(".tmp")
        payload = {
            "active_reference_id": self.active_reference_id,
            "references": self.references,
        }
        with temporary_path.open("w", encoding="utf-8") as catalog_file:
            json.dump(payload, catalog_file, ensure_ascii=False, indent=2)
        os.replace(temporary_path, self.catalog_path)

    def _load_or_seed(self) -> None:
        if self.catalog_path.is_file():
            try:
                catalog = json.loads(self.catalog_path.read_text(encoding="utf-8"))
                references = catalog["references"]
                if not isinstance(references, list):
                    raise ValueError("Trường references phải là một danh sách.")
                self.references = references
                self.active_reference_id = catalog.get("active_reference_id")
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                raise RuntimeError(
                    f"Không đọc được danh mục mạch tham chiếu {self.catalog_path}: {exc}"
                ) from exc
            if self.active_reference_id is not None:
                self._activate(self.active_reference_id, persist=False)
            return

        if (
            self.pipeline.reference_data
            and self.pipeline.reference_components
            and self.pipeline.reference_image is not None
        ):
            reference_id = "default"
            paths = self._reference_paths(reference_id)
            paths["folder"].mkdir(parents=True, exist_ok=True)
            source_image = Path(self.pipeline.ref_image_path)
            if source_image.resolve() != paths["image"].resolve():
                shutil.copy2(source_image, paths["image"])
            paths["components"].write_text(
                json.dumps(self.pipeline.reference_data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            source_detected = source_image.with_name("reference_detected.jpg")
            if not source_detected.is_file():
                source_detected = self.legacy_detected_path
            if source_detected.is_file():
                shutil.copy2(source_detected, paths["annotated"])
            source_name = str(
                self.pipeline.reference_data.get("image", "Mạch chuẩn hiện tại")
            )
            self.references = [
                {
                    "id": reference_id,
                    "name": Path(source_name).stem or "Mạch chuẩn hiện tại",
                    "components_count": len(self.pipeline.reference_components),
                    "created_at": datetime.now(timezone.utc).isoformat(),
                }
            ]
            self.active_reference_id = reference_id
            self._persist()
            self._activate(reference_id, persist=False)
        else:
            self.references = []
            self.active_reference_id = None
            self._persist()

    def _activate(self, reference_id: str, persist: bool = True) -> dict[str, Any]:
        reference = next(
            (item for item in self.references if item["id"] == reference_id), None
        )
        if reference is None:
            raise KeyError(reference_id)
        paths = self._reference_paths(reference_id)
        if not paths["image"].is_file() or not paths["components"].is_file():
            raise RuntimeError(
                f"Dữ liệu của mạch tham chiếu '{reference['name']}' không đầy đủ."
            )

        previous_paths = (
            self.pipeline.ref_json_path,
            self.pipeline.ref_image_path,
        )
        previous_id = self.active_reference_id
        try:
            self.pipeline.ref_json_path = str(paths["components"])
            self.pipeline.ref_image_path = str(paths["image"])
            self.pipeline.load_reference()
            if self.pipeline.reference_image is None or not self.pipeline.reference_components:
                raise RuntimeError(
                    f"Không thể nạp dữ liệu mạch tham chiếu '{reference['name']}'."
                )
            self.active_reference_id = reference_id
            if persist:
                self._persist()
        except Exception:
            self.pipeline.ref_json_path, self.pipeline.ref_image_path = previous_paths
            self.pipeline.load_reference()
            self.active_reference_id = previous_id
            raise
        return reference

    def _clear_active_reference(self) -> None:
        self.active_reference_id = None
        self.pipeline.ref_json_path, self.pipeline.ref_image_path = (
            self.default_reference_paths
        )
        self.pipeline.reference_data = None
        self.pipeline.reference_components = []
        self.pipeline.reference_image = None
        self.pipeline.aligner = None

    def select(self, reference_id: str) -> dict[str, Any]:
        with self.operation_lock:
            try:
                return self._activate(reference_id)
            except KeyError as exc:
                raise LookupError(
                    f"Không tìm thấy mạch tham chiếu có mã '{reference_id}'."
                ) from exc

    def delete(self, reference_id: str) -> dict[str, Any]:
        with self.operation_lock:
            index = next(
                (
                    index
                    for index, reference in enumerate(self.references)
                    if reference["id"] == reference_id
                ),
                None,
            )
            if index is None:
                raise LookupError(
                    f"Không tìm thấy mạch tham chiếu có mã '{reference_id}'."
                )

            reference = self.references[index]
            was_active = self.active_reference_id == reference_id
            fallback = next(
                (
                    item
                    for item in self.references
                    if item["id"] != reference_id
                ),
                None,
            )

            if was_active and fallback is not None:
                self._activate(fallback["id"], persist=False)
            elif was_active:
                self._clear_active_reference()

            self.references.pop(index)
            try:
                self._persist()
            except Exception:
                self.references.insert(index, reference)
                if was_active:
                    self._activate(reference_id, persist=False)
                raise

            try:
                shutil.rmtree(self._reference_paths(reference_id)["folder"])
            except OSError as exc:
                raise RuntimeError(
                    f"Đã xóa mạch khỏi danh mục nhưng không thể xóa dữ liệu lưu trữ "
                    f"'{reference['name']}': {exc}"
                ) from exc

            return {
                "reference": reference,
                "active_reference": fallback if was_active else next(
                    (
                        item
                        for item in self.references
                        if item["id"] == self.active_reference_id
                    ),
                    None,
                ),
            }

    def create(
        self,
        image: np.ndarray,
        name: str,
        conf: float,
        imgsz: int,
    ) -> dict[str, Any]:
        with self.operation_lock:
            reference_id = uuid.uuid4().hex
            paths = self._reference_paths(reference_id)
            paths["folder"].mkdir(parents=True, exist_ok=False)
            previous_paths = (
                self.pipeline.ref_json_path,
                self.pipeline.ref_image_path,
            )
            previous_id = self.active_reference_id
            previous_references = list(self.references)
            try:
                self.pipeline.ref_json_path = str(paths["components"])
                self.pipeline.ref_image_path = str(paths["image"])
                result = self.pipeline.create_reference_from_image(
                    image, conf=conf, imgsz=imgsz
                )
                reference = {
                    "id": reference_id,
                    "name": name.strip() or f"Mạch tham chiếu {len(self.references) + 1}",
                    "components_count": result["components_count"],
                    "created_at": datetime.now(timezone.utc).isoformat(),
                }
                self.references.append(reference)
                self.active_reference_id = reference_id
                self._persist()
                return result | {"reference": reference}
            except Exception:
                self.pipeline.ref_json_path, self.pipeline.ref_image_path = previous_paths
                self.active_reference_id = previous_id
                self.references = previous_references
                try:
                    if previous_id is None:
                        self._clear_active_reference()
                    else:
                        self.pipeline.load_reference()
                finally:
                    shutil.rmtree(paths["folder"])
                raise

    def list_references(self) -> list[dict[str, Any]]:
        with self.operation_lock:
            return [dict(reference) for reference in self.references]

    def snapshot(self) -> dict[str, Any]:
        with self.operation_lock:
            return {
                "active_reference_id": self.active_reference_id,
                "references": [dict(reference) for reference in self.references],
            }

    def get_annotated_image_path(self) -> Path | None:
        with self.operation_lock:
            if self.active_reference_id is None:
                return None
            path = self._reference_paths(self.active_reference_id)["annotated"]
            return path if path.is_file() else None

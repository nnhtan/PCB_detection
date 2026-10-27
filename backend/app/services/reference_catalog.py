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

    def select(self, reference_id: str) -> dict[str, Any]:
        with self.operation_lock:
            try:
                return self._activate(reference_id)
            except KeyError as exc:
                raise LookupError(
                    f"Không tìm thấy mạch tham chiếu có mã '{reference_id}'."
                ) from exc

    @staticmethod
    def _remove_tree(path: Path) -> None:
        def _on_error(func, p, exc_info):
            import stat

            try:
                os.chmod(p, stat.S_IWRITE)
                func(p)
            except Exception:
                pass

        shutil.rmtree(path, onerror=_on_error)

    def delete(self, reference_id: str) -> dict[str, Any]:
        with self.operation_lock:
            reference = next(
                (item for item in self.references if item["id"] == reference_id),
                None,
            )
            if reference is None:
                raise LookupError(
                    f"Không tìm thấy mạch tham chiếu có mã '{reference_id}'."
                )

            paths = self._reference_paths(reference_id)
            references_dir = self.references_dir.resolve()
            folder = paths["folder"]
            if folder.is_symlink() or (
                folder.exists() and folder.resolve().parent != references_dir
            ):
                raise RuntimeError("Đường dẫn dữ liệu mạch tham chiếu không hợp lệ.")

            remaining = [
                item for item in self.references if item["id"] != reference_id
            ]
            next_reference = remaining[0] if remaining else None
            was_active = self.active_reference_id == reference_id
            previous_state = {
                "references": self.references,
                "active_reference_id": self.active_reference_id,
                "reference_data": self.pipeline.reference_data,
                "reference_components": self.pipeline.reference_components,
                "reference_image": self.pipeline.reference_image,
                "aligner": self.pipeline.aligner,
                "ref_json_path": self.pipeline.ref_json_path,
                "ref_image_path": self.pipeline.ref_image_path,
            }
            staged_folder = folder.with_name(
                f".{folder.name}.deleting-{uuid.uuid4().hex}"
            )

            try:
                if folder.exists():
                    os.replace(folder, staged_folder)
                if was_active:
                    if next_reference is not None:
                        self._activate(next_reference["id"], persist=False)
                    else:
                        self.pipeline.reference_data = None
                        self.pipeline.reference_components = []
                        self.pipeline.reference_image = None
                        self.pipeline.aligner = None
                        self.active_reference_id = None
                self.references = remaining
                if was_active:
                    self.active_reference_id = (
                        next_reference["id"] if next_reference is not None else None
                    )
                else:
                    self.active_reference_id = previous_state["active_reference_id"]
                self._persist()
            except Exception:
                self.references = previous_state["references"]
                self.active_reference_id = previous_state["active_reference_id"]
                self.pipeline.reference_data = previous_state["reference_data"]
                self.pipeline.reference_components = previous_state[
                    "reference_components"
                ]
                self.pipeline.reference_image = previous_state["reference_image"]
                self.pipeline.aligner = previous_state["aligner"]
                self.pipeline.ref_json_path = previous_state["ref_json_path"]
                self.pipeline.ref_image_path = previous_state["ref_image_path"]
                if staged_folder.exists() and not folder.exists():
                    os.replace(staged_folder, folder)
                raise

            cleanup_warning = None
            if staged_folder.exists():
                try:
                    self._remove_tree(staged_folder)
                except OSError as exc:
                    cleanup_warning = (
                        "Mạch đã được xóa khỏi danh mục nhưng chưa dọn được "
                        f"thư mục dữ liệu: {exc}"
                    )

            return {
                "reference": dict(reference),
                "was_active": was_active,
                "active_reference_id": self.active_reference_id,
                "active_reference": (
                    dict(next_reference)
                    if was_active and next_reference is not None
                    else None
                    if was_active
                    else next(
                        (
                            dict(item)
                            for item in remaining
                            if item["id"] == self.active_reference_id
                        ),
                        None,
                    )
                ),
                "cleanup_warning": cleanup_warning,
            }

    def delete_unused(self) -> dict[str, Any]:
        with self.operation_lock:
            unused = [
                item
                for item in self.references
                if item["id"] != self.active_reference_id
            ]
            if not unused:
                return {
                    "deleted_count": 0,
                    "deleted_references": [],
                    "warnings": [],
                }

            deleted = []
            warnings = []
            references_dir = self.references_dir.resolve()
            staged_folders: list[Path] = []

            for item in unused:
                ref_id = item["id"]
                paths = self._reference_paths(ref_id)
                folder = paths["folder"]
                if folder.is_symlink() or (
                    folder.exists() and folder.resolve().parent != references_dir
                ):
                    warnings.append(
                        f"Bỏ qua mạch '{item['name']}': đường dẫn không hợp lệ."
                    )
                    continue

                staged_folder = folder.with_name(
                    f".{folder.name}.deleting-{uuid.uuid4().hex}"
                )
                try:
                    if folder.exists():
                        os.replace(folder, staged_folder)
                        staged_folders.append(staged_folder)
                    deleted.append(item)
                except Exception as exc:
                    warnings.append(
                        f"Không thể di dời thư mục của mạch '{item['name']}': {exc}"
                    )
                    if staged_folder.exists() and not folder.exists():
                        os.replace(staged_folder, folder)

            deleted_ids = {item["id"] for item in deleted}
            previous_references = list(self.references)
            self.references = [
                item for item in self.references if item["id"] not in deleted_ids
            ]
            try:
                self._persist()
            except Exception:
                self.references = previous_references
                for staged_folder in staged_folders:
                    original_name = staged_folder.name.split(".deleting-")[0].lstrip(".")
                    orig = staged_folder.with_name(original_name)
                    if staged_folder.exists() and not orig.exists():
                        os.replace(staged_folder, orig)
                raise

            for staged_folder in staged_folders:
                if staged_folder.exists():
                    try:
                        self._remove_tree(staged_folder)
                    except OSError as exc:
                        warnings.append(
                            f"Chưa dọn được thư mục tạm {staged_folder.name}: {exc}"
                        )

            return {
                "deleted_count": len(deleted),
                "deleted_references": [dict(item) for item in deleted],
                "warnings": warnings,
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
                try:
                    self.pipeline.load_reference()
                finally:
                    self.active_reference_id = previous_id
                    self.references = previous_references
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

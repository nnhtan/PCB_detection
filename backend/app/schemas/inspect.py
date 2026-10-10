from pydantic import BaseModel, Field


class ComponentDetection(BaseModel):
    class_id: int
    label: str
    confidence: float
    x: float
    y: float
    width: float
    height: float


class InspectionResult(BaseModel):
    status: str
    detections: list[ComponentDetection]
    missing_components: list[str] = Field(default_factory=list)
    unexpected_components: list[str] = Field(default_factory=list)

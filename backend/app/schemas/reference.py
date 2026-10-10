from pydantic import BaseModel, Field


class ReferenceComponent(BaseModel):
    id: int
    class_id: int
    label: str = Field(alias="class")
    confidence_reference: float
    x: float
    y: float


class ReferenceBoard(BaseModel):
    image: str
    image_width: int
    image_height: int
    components_count: int
    components: list[ReferenceComponent]

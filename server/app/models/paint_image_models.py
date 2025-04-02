from pydantic import BaseModel, ValidationError, validator
from typing import List, Optional

# Pydantic models
class Modification(BaseModel):
    x: int
    y: int
    isEraser: bool
    color: Optional[str] = None
    shape: str  # 'square' or 'circle'
    size: int

    @validator('shape')
    def validate_shape(cls, v, values):
        if values.get('isEraser', False):
            if v != 'square':
                raise ValueError("If isEraser is true, shape must be 'square'")
        else:
            if v not in ['square', 'circle']:
                raise ValueError("Shape must be 'square' or 'circle'")
        return v

    @validator('color', always=True)
    def validate_color(cls, v, values):
        if not values.get('isEraser', False) and not v:
            raise ValueError("Color must be provided when isEraser is false")
        return v

class Layer(BaseModel):
    modifications: List[Modification]
from typing import List
from enum import Enum
from pydantic import BaseModel
from typing import Dict
from typing import Optional

class LedParameter(BaseModel):
    nomal: int
    packed: int

class CPUUsageResponse(BaseModel):
    cpu_usage: float
    announcement: str

class PerImageParameter(BaseModel):
    image: List[List[int]]
    height: float
    width: float
    area: float
    peri: float
    area_for_frontend: int
    perimeter_for_frontend: int
    x: float
    y: float
    distance: List[float]
    skeleton_length: float
    intersection_count3: int
    intersection_count4: int
    intersection_count5: int
    intersection_count6: int
    endpoints_count: int

class PredictedAndActualLEDNumber(BaseModel):
    predicted: List[int]
    actual: List[int]


class PerImageParameterWithModel(PerImageParameter):
    luminous_model: int


class PerImageParameterWithLED(PerImageParameterWithModel):
    led: int


class WholeImageParameterWithLED(BaseModel):
    height: int
    width: int
    scale_ratio: float
    scale_ratio_for_frontend_scale: float
    perimages: Dict[int, PerImageParameterWithLED]

class Announcement(BaseModel):
    announcement: str


class WholeImageParameter(BaseModel):
    height: int
    width: int
    scale_ratio: float
    scale_ratio_for_frontend_scale: float
    url: str
    perimages: Dict[int, PerImageParameter]


class LuminousModel(Enum):
    HYOMEN = 1
    NEON = 2
    URAMEN = 3
    URASOKUMEN = 4
    SOKUMEN = 5
    TYPEA = 6


class WholeImagePredictParameter(BaseModel):
    height: int
    width: int
    scale_ratio: float
    scale_ratio_for_frontend_scale: float
    perimages: Dict[int, PerImageParameterWithModel]


class ImageData(BaseModel):
    base64_image: Optional[str] = None
    url: Optional[str] = None
    whole_hight_mm: float


class ImageHeightAndWidth(BaseModel):
    height: int
    width: int


class ImageRequest(BaseModel):
    image_dictionary: Dict[str, List[List[int]]]
    scale_ratio: float


class LearningDataInformation(BaseModel):
    number: int
    number_of_under_5: int
    number_of_under_5_to_10: int
    number_of_over_10: int


class NumberOfLearningData(BaseModel):
    hyomen: LearningDataInformation
    uramen: LearningDataInformation
    sokumen: LearningDataInformation
    typea: LearningDataInformation
    neon: LearningDataInformation
    urasokumen: LearningDataInformation

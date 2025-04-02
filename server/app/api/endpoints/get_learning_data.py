import logging
import os

from fastapi import APIRouter, HTTPException
import pandas as pd
from app.models.led_output import ImageHeightAndWidth, ImageRequest, NumberOfLearningData, LearningDataInformation
from app.services.predict_led import mape_scores

logger = logging.getLogger(__name__)
router = APIRouter()

file_map = {
    "hyomen": "predict_database.tsv",
    "neon": "predict_database_neon.tsv",
    "uramen": "predict_database_uramen.tsv",
    "sokumen": "predict_database_sokumen.tsv",
    "typea": "predict_database_typea.tsv",
    "urasokumen": "predict_database_urasokumen.tsv",
}

def get_unique_pj_count(condition: str):
    try:
        df = pd.read_csv(os.path.join(os.getcwd(), 'app', 'services', 'led_database', file_map[condition]), sep="\t", header=0)
        return len(df)
    except Exception as e:
        logger.error(f"Error loading file for condition {condition}: {e}")
        raise

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))

def load_mape_scores_from_file(filename="mape_scores.txt"):
    mape_scores = {}
    with open(os.path.join(parent_dir, 'services', 'led_database', filename), 'r') as f:
        for line in f:
            # 各行を "key: value" 形式として扱う
            try:
                key, value = line.strip().split(": ")
                mape_scores[key] = float(value)
            except ValueError:
                pass  # floatに変換できない場合は無視
    return mape_scores

mape_scores = load_mape_scores_from_file()

@router.post("/get_learning_data/", response_model=NumberOfLearningData)
async def get_learning_data():
    try:
        # Get unique 'pj' counts for each dataset
        hyomen_count = get_unique_pj_count("hyomen")
        uramen_count = get_unique_pj_count("uramen")
        sokumen_count = get_unique_pj_count("sokumen")
        typea_count = get_unique_pj_count("typea")
        urasokumen_count = get_unique_pj_count("urasokumen")
        neon_count = get_unique_pj_count("neon")

        return NumberOfLearningData(
            hyomen=LearningDataInformation(number=hyomen_count, number_of_under_5=mape_scores.get("hyomen_0_5", 0),
                                           number_of_under_5_to_10=mape_scores.get("hyomen_5_10", 0),
                                           number_of_over_10=mape_scores.get("hyomen_10", 0)),
            uramen=LearningDataInformation(number=uramen_count, number_of_under_5=mape_scores.get("uramen_0_5", 0),
                                           number_of_under_5_to_10=mape_scores.get("uramen_5_10", 0),
                                           number_of_over_10=mape_scores.get("uramen_10", 0)),
            sokumen=LearningDataInformation(number=sokumen_count, number_of_under_5=mape_scores.get("sokumen_0_5", 0),
                                            number_of_under_5_to_10=mape_scores.get("sokumen_5_10", 0),
                                            number_of_over_10=mape_scores.get("sokumen_10", 0)),
            typea=LearningDataInformation(number=typea_count, number_of_under_5=mape_scores.get("typea_0_5", 0),
                                          number_of_under_5_to_10=mape_scores.get("typea_5_10", 0),
                                          number_of_over_10=mape_scores.get("typea_10", 0)),
            urasokumen=LearningDataInformation(number=urasokumen_count, number_of_under_5=mape_scores.get("urasokumen_0_5", 0),
                                          number_of_under_5_to_10=mape_scores.get("urasokumen_5_10", 0),
                                          number_of_over_10=mape_scores.get("urasokumen_10", 0)),
            neon=LearningDataInformation(number=neon_count, number_of_under_5=mape_scores.get("neon_0_5", 0),
                                          number_of_under_5_to_10=mape_scores.get("neon_5_10", 0),
                                          number_of_over_10=mape_scores.get("neon_10", 0))
        )

    except Exception as e:
        logger.error(f"Error Get Learning deta Request: {e}")
        logger.exception(e)
        raise HTTPException(status_code=500, detail=str(e))
import pandas as pd
from snorkel.labeling.model import LabelModel


class BaseDetector:
    def __init__(self) -> None:
        pass

    def is_drift(self, model: LabelModel, validation_data: pd.DataFrame) -> bool:
        print("This detector shouldn't be used!! It will always return True.")
        return True
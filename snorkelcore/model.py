from .lflibrary import LabelingFunctionLibrary
from typing import List, Tuple, Union
from snorkel.labeling import PandasLFApplier
from snorkel.labeling.model import LabelModel
import pandas as pd
import numpy as np

class SnorkelServeModel:
    def __init__(
            self,
            label_func_lib: LabelingFunctionLibrary,
            data_ingestor: 'BaseIngestor',
            cardinality: int,
            batch_size: int=50,
            train_epochs: int=500,
            log_freq: int=100
        ) -> None:
        self.label_func_lib = label_func_lib
        self.data_ingestor = data_ingestor
        self.cardinality = cardinality
        self.model = None
        self.applier = None
        self.cache = []
        self.predictions = []
        self.batch_size = batch_size
        self.train_epochs = train_epochs
        self.log_freq = log_freq
        self.is_running = False
    
    def serve(self) -> List[int]:
        while self.is_running:
            # Keep append data if do not achieve batch size
            if len(self.cache) < self.batch_size:
                self.cache.append(next(self.data_ingestor))
                continue

            # Train model if we don't have a model
            if self.model is None:
                self.train_model()

            # Append prediction result to predictions
            self.predictions.append(self.predict())

            # Clear the cache
            self.cache = []

    def train_model(self) -> None:
        lfs = self.label_func_lib.get_all()
        self.applier = PandasLFApplier(lfs=lfs)
        df = pd.DataFrame(self.cache)
        L_train = self.applier.apply(df=df)
        self.model = LabelModel(cardinality=self.cardinality)
        self.model.fit(L_train=L_train, n_epochs=self.train_epochs, log_freq=self.log_freq)

    def predict(self) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        df = pd.DataFrame(self.cache)
        L = self.applier.apply(df)
        return self.model.predict(L=L)
    
    def run(self) -> None:
        self.is_running = True
        self.serve()
    
    def stop(self) -> None:
        self.is_running = False
    
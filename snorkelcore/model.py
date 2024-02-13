from .lflibrary import LabelingFunctionLibrary
from typing import List, Dict
from snorkel.labeling import PandasLFApplier
from snorkel.labeling.model import LabelModel
from .driftdetector.detectors import BaseDetector
import pandas as pd

class SnorkelServeModel:
    def __init__(
            self,
            label_func_lib: LabelingFunctionLibrary,
            data_ingestor: 'BaseIngestor',
            cardinality: int,
            drift_detector: BaseDetector,
            batch_size: int=50,
            train_epochs: int=500,
            log_freq: int=100,
            label_map: Dict[int, str]=None,
            drift_check_freq: int=20
        ) -> None:
        self.label_func_lib = label_func_lib
        self.data_ingestor = data_ingestor
        self.cardinality = cardinality
        self.drift_detector = drift_detector
        self.model = None
        self.applier = None
        self.cache = []
        self.predictions = []
        self.batch_size = batch_size
        self.train_epochs = train_epochs
        self.log_freq = log_freq
        self.label_map = label_map
        self.is_running = False
        self.drift_check_freq = drift_check_freq
        self.serve_cnt = 0
    
    def flush(self) -> None:
        self.serve_cnt = (self.serve_cnt + 1) % self.drift_check_freq
        if self.serve_cnt == 0:
            # Check for model drift every `drift_check_freq` requests
            is_drifted = self.drift_detector.is_drift(self.model, self.get_cache_df())
            if is_drifted:
                print("Drift detected! Re-training the model...")
                self.train_model()

        predict = self.predict()
        self.predictions.append(predict)

    def serve(self) -> None:
        while self.is_running:
            # Keep append data if do not achieve batch size
            if len(self.cache) < self.batch_size:
                try:
                    data = next(self.data_ingestor)
                    self.cache.append(data[1])
                except StopIteration:
                    print("All data has been read, stop...")
                    if self.cache:
                        self.flush()
                    return
                continue

            # Train model if we don't have a model
            if self.model is None:
                self.train_model()

            # Flush cache
            self.flush()

            # Clear the cache
            self.cache = []

    def train_model(self) -> None:
        lfs = self.label_func_lib.get_all()
        self.applier = PandasLFApplier(lfs=lfs)
        df = self.get_cache_df()
        L_train = self.applier.apply(df=df)
        self.model = LabelModel(cardinality=self.cardinality)
        self.model.fit(L_train=L_train, n_epochs=self.train_epochs, log_freq=self.log_freq)

    def get_cache_df(self) -> pd.DataFrame:
        return pd.concat(self.cache, axis=1).transpose()

    def predict(self) -> pd.DataFrame:
        df = self.get_cache_df()
        L = self.applier.apply(df)
        df['label'] = self.model.predict(L=L)
        if self.label_map is not None:
            df['label'] = df['label'].map(self.label_map)
        return df
    
    def run(self) -> None:
        self.is_running = True
        self.serve()
    
    def stop(self) -> None:
        self.is_running = False
    
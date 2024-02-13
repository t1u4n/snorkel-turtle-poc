from .lflibrary import LabelingFunctionLibrary
from typing import List, Dict
from snorkel.labeling import PandasLFApplier
from snorkel.labeling.model import LabelModel
import pandas as pd

class SnorkelServeModel:
    def __init__(
            self,
            label_func_lib: LabelingFunctionLibrary,
            data_ingestor: 'BaseIngestor',
            cardinality: int,
            batch_size: int=50,
            train_epochs: int=500,
            log_freq: int=100,
            label_map: Dict[int, str]=None
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
        self.label_map = label_map
        self.is_running = False
    
    def flush(self) -> None:
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
        df = pd.concat(self.cache, axis=1).transpose()
        L_train = self.applier.apply(df=df)
        self.model = LabelModel(cardinality=self.cardinality)
        self.model.fit(L_train=L_train, n_epochs=self.train_epochs, log_freq=self.log_freq)

    def predict(self) -> pd.DataFrame:
        df = pd.concat(self.cache, axis=1).transpose()
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
    
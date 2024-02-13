class BaseIngestor:
    def __init__(self) -> None:
        pass

    def __next__(self) -> str:
        return "not implemented"

import pandas as pd   
class CSVFileIngestor(BaseIngestor):
    def __init__(self, file_path: str) -> None:
        df = pd.read_csv(file_path)
        self.iter = df.iterrows()
        self.col_names = df.columns.to_list()
    
    def __next__(self) -> str:
        return next(self.iter)
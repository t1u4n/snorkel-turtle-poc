from typing import List, Any
import pandas as pd
import os

class BaseLoader:
    def __init__(self) -> None:
        print("This is a base data loader that should not be used.")

    def flush(self, buf: List[Any]) -> None:
        print("Not implemented")


class CSVLoader(BaseLoader):
    def __init__(self, filename: str) -> None:
        self.filename = filename
        super().__init__()

    def flush(self, buf: List[pd.Series]) -> None:
        """Flush the buffer to target csv file"""
        df = pd.concat(buf)
        if not os.path.exists(self.filename):
            df.to_csv(self.filename, index=False)
        else:
            df.to_csv(self.filename, mode='a', index=False, header=False)
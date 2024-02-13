from ingestor.ingestors import CSVFileIngestor
from snorkelcore.lflibrary import LabelingFunctionLibrary
from snorkelcore.model import SnorkelServeModel
from snorkel.labeling import labeling_function
import pandas as pd

@labeling_function(name="is_positive")
def positive_sentiment(x):
    return 1 if 'good' in x.text else 0

@labeling_function(name="comment_class")
def comment_class(x):
    if 'book' in x.text:
        return 1
    return 0

@labeling_function(name="noise")
def noise(x):
    return 1 if 'do' in x.text else 0

def run():
    ingestor = CSVFileIngestor('resources/data/data.csv')
    
    lfs = [positive_sentiment, comment_class, noise]
    lib = LabelingFunctionLibrary()
    for lf in lfs:
        lib.register(lf.name, lf)
    label_map = {1: "Good Book", 0: "Bad Movie", -1: "N/A"}

    model = SnorkelServeModel(
        label_func_lib=lib,
        data_ingestor=ingestor,
        cardinality=2,
        batch_size=3,
        label_map=label_map
    )

    try:
        model.run()
    finally:
        # Write predictions
        pd.concat(model.predictions).to_csv("output/predictions.csv", index=False)
        model.stop()
    

if __name__=="__main__":
    run()
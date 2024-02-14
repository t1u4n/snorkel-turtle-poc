from ingestor.ingestors import CSVFileIngestor
from loader.loaders import CSVLoader
from snorkelcore.lflibrary import LabelingFunctionLibrary
from snorkelcore.model import SnorkelServeModel
from snorkelcore.driftdetector.detectors import BaseDetector
from snorkel.labeling import labeling_function

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

    loader = CSVLoader('output/predictions.csv')
    
    lfs = [positive_sentiment, comment_class, noise]
    lib = LabelingFunctionLibrary()
    for lf in lfs:
        lib.register(lf.name, lf)
    
    label_map = {1: "Good Book", 0: "Bad Movie", -1: "N/A"}

    drift_detector = BaseDetector()

    model = SnorkelServeModel(
        label_func_lib=lib,
        data_ingestor=ingestor,
        data_loader=loader,
        cardinality=2,
        drift_detector=drift_detector,
        batch_size=1,
        drift_check_freq=1,
        load_batch_size=1,
        label_map=label_map
    )

    try:
        model.run()
    finally:
        model.stop()
    

if __name__=="__main__":
    run()
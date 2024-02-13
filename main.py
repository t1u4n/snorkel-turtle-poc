from ingestor.ingestors import BaseIngestor

def run():
    base_ingestor = BaseIngestor()
    print(base_ingestor.next())

if __name__=="__main__":
    run()
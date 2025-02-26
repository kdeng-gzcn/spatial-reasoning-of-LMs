class MetricTemplate:

    def __init__(self, **kwargs):

        self.result_dict = []
   
    def _extract_info(self):
    
        raise NotImplementedError("Subclasses must implement the _extract_info method.")
        
    def process_conclusion(self):

        raise NotImplementedError("Subclasses must implement the process_conversations method.")    
    
    def evaluate(self):

        raise NotImplementedError("Subclasses must implement the evaluate method.")

if __name__ == "__main__":

    pass

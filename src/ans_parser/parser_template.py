class ParserTemplate:
    def __init__(self):
        pass
   
    def _extract_info(self):
        raise NotImplementedError("Subclasses must implement the _extract_info method.")
        
    def process_conclusion(self):
        raise NotImplementedError("Subclasses must implement the process_conversations method.")    

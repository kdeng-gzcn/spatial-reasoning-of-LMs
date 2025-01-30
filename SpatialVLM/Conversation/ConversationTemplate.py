# load prompter
from SpatialVLM.Prompter.utils import load_prompter # eyeprompt in dataloader
# load dataset
from SpatialVLM.Dataset.utils import load_dataset # dataset
from torch.utils.data import DataLoader # dataloader
# load metrics
from SpatialVLM.Metrics.utils import _
# load model
from SpatialVLM.Model.utils import load_model

class ConversationTemplate():

    def __init__(self, VLM_id=None, LLM_id=None, datapath=None):

        # 1. data
        data_path = datapath

        dataset = SevenScenesImageDataset(root_dir=data_path)
        self.dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

        # 2. VLM
        if VLM_id == "remyxai/SpaceLLaVA":
            model = VLMTemplate.SpaceLLaVA()
            model() # load cache

        if VLM_id == "llava-hf/llava-v1.6-mistral-7b-hf":
            model = VLMTemplate.LlavaNextVLM()
            model.load_model(VLM_id) # load cache

        if VLM_id == "HuggingFaceM4/idefics2-8b":
            model = VLMTemplate.Idefics2VLM()
            model.load_model(VLM_id)

        if VLM_id == "microsoft/Phi-3.5-vision-instruct":
            model = VLMTemplate.Phi3VLM()
            model.load_model(VLM_id)

        self.VLM = model

        # 3. LLM
        if LLM_id == "meta-llama/Meta-Llama-3-8B-Instruct":
            model = HuggingFaceLLM(LLM_id)
            model() # load cache
               
        self.LLM = model

        # 4. prompter
        self.brain_prompt = BrainPrompt()

        # 5. writer
        self.writer = Writer(self.VLM, self.LLM)

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

        dataset = load_dataset("7 Scenes", data_root_dir=data_path)
        self.dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

        # 2.VLM
        self.VLM_id = VLM_id
        self.VLM = load_model("Phi 3.5")
        self.VLM._load_weight()

        # 3. LLM
        self.LLM_id = LLM_id
        self.LLM = load_model("llama")
        self.LLM._load_weight()

        # 4. prompter
        self.start_prompter = load_prompter("Begin")
        self.LLM_prompter = load_prompter("Brain")

    def __call__(self):

        raise NotImplementedError()

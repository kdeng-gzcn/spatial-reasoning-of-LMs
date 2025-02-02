# load prompter
from SpatialVLM.Prompter.utils import load_prompter # eyeprompt in dataloader
# load dataset
from SpatialVLM.Dataset.utils import load_dataset # dataset
from torch.utils.data import DataLoader # dataloader
# load metrics
# from SpatialVLM.Metrics.utils import _
# load model
from SpatialVLM.Model.utils import load_model

class ConversationTemplate():

    def __init__(self, **kwargs):

        self.VLM_id = kwargs.get("VLM_id", None)
        self.LLM_id = kwargs.get("LLM_id", None)
        data_path = kwargs.get("datapath", None)
        self.subset = kwargs.get("subset", None)

        # 1. data
        dataset = load_dataset("7 Scenes", data_root_dir=data_path, subset=self.subset)
        self.dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=lambda x: x)

        # 2.VLM
        self.VLM = load_model("Phi 3.5")
        self.VLM._load_weight()

        # 3. LLM
        self.LLM = load_model("llama")
        self.LLM._load_weight()

        # 4. prompter
        self.start_prompter = load_prompter("Begin")
        self.LLM_prompter = load_prompter("Brain")

    def __call__(self):

        raise NotImplementedError()

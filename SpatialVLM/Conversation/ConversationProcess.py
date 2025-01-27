import sys
sys.path.append('./')

# load prompter
from SpatialVLM.Prompter.Prompter import BrainPrompt, StartPrompt # eyeprompt in dataloader
# load dataset
from SpatialVLM.utils.dataset.dataloader_7Scenes import SevenScenesImageDataset # dataset
from torch.utils.data import DataLoader # dataloader
# load writer
from SpatialVLM.utils.Writer import Writer # writer
# load metric

# load model
from SpatialVLM.Model import VLMTemplate # VLM
from SpatialVLM.Model.LLMTemplate import HuggingFaceLLM # LLM

class ConversationTemplate():
    """
    Alg.py: parse all hyper params and combinate all tools (handle str)
    """
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

class Conversations_Single_Image(ConversationTemplate):
    """
    
    This is for single-image conversation

    """
    def __init__(self, VLM_id=None, LLM_id=None, datapath=None):
        # init all built-in tools
        super().__init__(VLM_id=VLM_id, LLM_id=LLM_id, datapath=datapath)
    
    def __call__(self, len_conversation=1):

        # setting up important part
        dataloader = self.dataloader # dataloader
        VLM = self.VLM # eyes after loading
        LLM = self.LLM # brain after loading
        LLM_prompter = self.brain_prompt # prompter to help brain
        writer = self.writer # log with 1. start 2. each step

        for batch in dataloader:
            source_images, source_prompts, source_image_names, target_images, target_prompts, target_image_names = batch

            # take 1 sample from 1 batch (edit dataloader later)
            for (source_image, source_prompt, source_image_name, target_image, target_prompt, target_image_name) in zip(source_images, source_prompts, source_image_names, target_images, target_prompts, target_image_names):

                # NOW WE GET A PAIR OF IMAGES
                # 0. max conversation length and clear LLM History in each step
                LLM.clearhistory()
                len_conversation = len_conversation

                # 1. load start prompt for task description
                task_prompter = StartPrompt()
                task_prompt = task_prompter(mode="single")
                # 1.1 get the questions from LLM after it understand our task
                LLM_Questions_for_Source = LLM.pipeline(task_prompt)

                ############################ log start
                writer.write_at_start_single(
                    source_image_name=source_image_name,
                    target_image_name=target_image_name,
                    task_prompt=task_prompt
                )
                ############################ log start

                # 2. start conversation
                for idx in range(len_conversation): # for loop for max length or stop condition

                    if idx:
                        # if idx > 0 and LLM is not satisfied, then extract new questions for source
                        LLM_Questions_for_Source = LLM_Answer_for_round_idx

                    # a. get answers from VLM given questions for source from LLM
                    VLM_Answer_for_Sourse = VLM.pipeline(source_image, LLM_Questions_for_Source)
                    VLM_Answer_for_Sourse = LLM_prompter.answer_prompter_single(VLM_Answer_for_Sourse, source=True)

                    # b. get questions for target
                    LLM_Questions_for_Target = LLM.pipeline(VLM_Answer_for_Sourse)

                    # c. get answers from VLM given questions for target from LLM
                    VLM_Answer_for_Target = VLM.pipeline(source_image, LLM_Questions_for_Target)
                    VLM_Answer_for_Target = LLM_prompter.answer_prompter_single(VLM_Answer_for_Target, source=False)

                    # d. now we want to get answers from LLM to see if he understand the images really
                    LLM_Answer_for_round_idx = LLM.pipeline(VLM_Answer_for_Target)

                    # *e. judge if stop the conversation

                    ############################ log end
                    writer.write_in_conversation_single(
                        question_for_source=LLM_Questions_for_Source, 
                        answer_for_source=VLM_Answer_for_Sourse,
                        question_for_target=LLM_Questions_for_Target,
                        answer_for_target=VLM_Answer_for_Target,
                        llm_answer=LLM_Answer_for_round_idx,
                        idx=idx
                    )
                    ############################ log end
                
                # print("history:", LLM.full_history)
                # print("messages:", LLM.messages)

        writer.close()

class Conversations_Pairwise_Image(ConversationTemplate):
    """
    
    This is for multi-image conversation

    """
    def __init__(self, VLM_id=None, LLM_id=None, datapath=None):
        # init all built-in tools
        super().__init__(VLM_id=VLM_id, LLM_id=LLM_id, datapath=datapath)
    
    def __call__(self, len_conversation=1):

        # setting up important part
        dataloader = self.dataloader # dataloader
        VLM = self.VLM # eyes after loading
        LLM = self.LLM # brain after loading
        LLM_prompter = self.brain_prompt # prompter to help brain
        writer = self.writer # log with 1. start 2. each step

        data_count = 0

        for batch in dataloader:
            # parse batch
            scene, seq, source_image_name, target_image_name, source_image, target_image, label_num, label_text, Description = batch

            # if data_count >= 5: # for small scale test
            #     break
            
            # data_count += 1

            # take 1 sample from 1 batch (edit dataloader later)
            for (scene, seq, source_image_name, target_image_name, source_image, target_image, label_num, label_text, Description) in zip(scene, seq, source_image_name, target_image_name, source_image, target_image, label_num, label_text, Description):

                # NOW WE GET A PAIR OF IMAGES and corresponding info
                # 0. Alg hyper params
                LLM.clearhistory() # for differnt pair images, clear up history in LLM
                len_conversation = len_conversation # max length of conversation
                images = [source_image, target_image] # fit in data structure

                # 1. load start prompt for task description
                task_prompter = StartPrompt()
                task_prompt = task_prompter(mode="pair")
                # 1.1 get the questions from LLM after it understand our task
                LLM_Questions_for_Both = LLM.pipeline(task_prompt)

                ############################ log start
                writer.write_at_start_pair(
                    scene=scene,
                    seq=seq,
                    source_image_name=source_image_name,
                    target_image_name=target_image_name,
                    task_prompt=task_prompt,
                    label_text=label_text
                )
                ############################ log start

                # 2. start conversation
                for idx in range(len_conversation): # for loop for max length or stop condition

                    if idx:
                        # if idx > 0 and LLM is not satisfied, then extract new questions for source
                        LLM_Questions_for_Both = LLM_Answer_for_round_idx

                    # a. get answers from VLM given questions for source from LLM
                    LLM_Questions_for_Both = "(Note that the first image is source image and the second one is target image) " + LLM_Questions_for_Both
                    VLM_Answer_for_Both = VLM.pipeline(images, LLM_Questions_for_Both)
                    VLM_Answer_for_Both = LLM_prompter.answer_prompter_pair(VLM_Answer_for_Both)

                    # b. now we want to get answers from LLM to see if he understand the images really
                    LLM_Answer_for_round_idx = LLM.pipeline(VLM_Answer_for_Both)

                    # *c. judge if stop the conversation

                    ############################ log each conversation end
                    writer.write_in_conversation_pair(
                        question_for_both=LLM_Questions_for_Both,
                        answer_for_both_VLM=VLM_Answer_for_Both,
                        llm_final_answer=LLM_Answer_for_round_idx,
                        idx=idx
                    )
                    ############################ log each conversation end
                
                # print("history:", LLM.full_history)
                # print("messages:", LLM.messages)

                # 3. evaluation PRINT!!!!!!!!!!!!!!!!!!!!!!!!!!
                label_true = label_text[4]
                

        writer.close() # close for all data

if __name__ == "__main__":
    
    pass

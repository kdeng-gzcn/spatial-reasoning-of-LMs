from .ConversationTemplate import ConversationTemplate

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

if __name__ == "__main__":
    
    pass

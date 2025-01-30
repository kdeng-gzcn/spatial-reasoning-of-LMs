from .ConversationTemplate import ConversationTemplate

class Conversations_Pairwise_Image(ConversationTemplate):

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
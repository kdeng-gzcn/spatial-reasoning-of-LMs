# import sys
# sys.path.append("./")

# from SpatialVLM.Model.VLMTemplate import 

# data_path = './data/images_conversation'  # set root data_path

# dataset = CustomImageDataset(root_dir=data_path)
# dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# for batch in dataloader:
#     source_images, source_prompts, source_image_names, target_images, target_prompts, target_image_names = batch
#     print(source_image_names, target_image_names)
# ####

# #### llava-Next
# # model_name = "llava-hf/llava-v1.6-mistral-7b-hf"

# # model = LlavaNextVLM()
# # model.load_model(model_id=model_name) # load cache

# # answer = model.pipeline(source_images[0], source_prompts[0])
# # print("HF: ", answer)

# #### idefics2
# # model_name = "HuggingFaceM4/idefics2-8b"

# # model = Idefics2VLM()
# # model.load_model(model_id=model_name) # load cache

# # images = [source_images[0], target_images[0]]

# # answer = model.pipeline(images, "How many images do you see? What do you see in this pair of images, describe the main objects inside the images, and make comparisons between them.")

# # print("idefics2 output: ", answer)

# #### Phi3.5
# model_name = "microsoft/Phi-3.5-vision-instruct"

# model = Phi3VLM()
# model.load_model(model_id=model_name) # load cache

# images = [source_images[0], target_images[0]]

# answer = model.pipeline(images, "How many images do you see? What do you see in this pair of images, describe the main objects inside the images, and make comparisons between them.")

# print("Phi3.5 output: ", answer)
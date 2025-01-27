import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms

import sys
sys.path.append('./')

from SpatialVLM.Prompter.Prompter import EyePrompt

class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform=transforms.ToTensor()):
        self.root_dir = root_dir
        self.transform = transform
        self.image_pairs = self._load_image_pairs()

        # call prompt template
        self.prompt_generator = EyePrompt()

    def _load_image_pairs(self):
        image_pairs = []
        # iterate images_conversation, every imagesX folder
        for folder_name in os.listdir(self.root_dir):
            image_folder = os.path.join(self.root_dir, folder_name)
            source_folder = os.path.join(image_folder, 'source')
            target_folder = os.path.join(image_folder, 'target')

            # get source and target folder image files
            source_images = sorted(os.listdir(source_folder))
            target_images = sorted(os.listdir(target_folder))

            # make source and target a pair
            for src_img, tgt_img in zip(source_images, target_images):
                source_path = os.path.join(source_folder, src_img)
                target_path = os.path.join(target_folder, tgt_img)
                image_pairs.append((source_path, target_path))

        return image_pairs

    def _generate_prompt(self, is_source=True):
        """
        Maybe Useless
        """
        # if is_source:
        #     return f"This is a source image: {image_name}"
        # else:
        #     return f"This is a target image: {image_name}"

        if is_source:
            prompt = self.prompt_generator()
            return prompt
        else:
            eye = EyePrompt()
            prompt = eye()
            return prompt

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        source_path, target_path = self.image_pairs[idx]
        source_image = Image.open(source_path).convert("RGB")
        target_image = Image.open(target_path).convert("RGB")

        # extract image name!!!
        source_image_name = os.path.basename(source_path)
        target_image_name = os.path.basename(target_path)

        # generate prompt given images
        source_prompt = self._generate_prompt(is_source=True)
        target_prompt = self._generate_prompt(is_source=False)

        # image pre-processing (ToTensor())
        if self.transform:
            source_image = self.transform(source_image)
            target_image = self.transform(target_image)

        return source_image, source_prompt, source_image_name, target_image, target_prompt, target_image_name

if __name__ == "__main__":

    data_path = './data/images_conversation'  # set root data_path

    dataset = CustomImageDataset(root_dir=data_path)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    for batch in dataloader:
        source_images, source_prompts, source_image_names, target_images, target_prompts, target_image_names = batch
        print(source_images, source_prompts, source_image_names, target_images, target_prompts, target_image_names)

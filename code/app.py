import os 
from tqdm import tqdm
from loguru import logger
import numpy as np

from typing import List
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import torch
import torch.nn.functional as F
import clip
from PIL import Image


class CLIP:
    def __init__(self, model_name: str) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name

    def load_image_model(self):
        device = self.device
        model, preprocess = clip.load(self.model_name, device=device)
        return model, preprocess

    def preprocess_image(self, img_path):
        _, preprocess = self.load_image_model()
        image = preprocess(Image.open(img_path)).unsqueeze(0).to(self.device)
        return image

    def get_text_similarity_based_on_single_inp_image(self, image_path: str, candidates_name: List[str], model, preprocess) -> dict:

        # loads and return array info of image
        image = self.preprocess_image(image_path)
        text = clip.tokenize(candidates_name).to(self.device)

        # logging no grads 
        with torch.no_grad():
            # just for getting embeds [ not necessary to pass in model arch ]
            image_features = model.encode_image(image)
            text_features = model.encode_text(text)
        
            logits_per_image, logits_per_text = model(image, text)
            # breakpoint()
            probs = logits_per_image.softmax(dim=-1)
            print(probs)
            index = torch.argmax(probs, dim=1)

        return {"index": index, "probs": probs, "category": candidates_name[index]}

    def get_image_to_image_similarity(self, inp_image_path, ref_image_dir_path, model, preprocess) -> dict:

        all_img_dir_embeds = []
        all_image_name = []
        with torch.no_grad():
            # cal img embeddings for both inp image and ref image dir path
            inp_embeds = model.encode_image(self.preprocess_image(inp_image_path)).tolist()
            # embeddings all img from ref docs 
            logger.info("Embedding all img from ref image dir ")
            for img in tqdm(os.listdir(ref_image_dir_path)):
                # if img in ['human_1.jpeg', 'human_2.jpeg']:
                if img.split(".")[-1] in ['jpeg', 'jpg', 'png']:
                  img_embeds = model.encode_image(self.preprocess_image(os.path.join(ref_image_dir_path, img)))
                  all_img_dir_embeds.append(img_embeds.tolist()[0])
                  all_image_name.append(img)
        
        # return inp_embeds, all_img_dir_embeds
        cos_sim = F.cosine_similarity(torch.tensor(inp_embeds), torch.tensor(all_img_dir_embeds), dim = 1)
        logger.debug("All img to ref image embeds cosine score ")
        logger.info(cos_sim)

        # get top k = 3 indices
        k = 3
        top_k = cos_sim.argsort(descending = True)[:k]

        # get embeds 
        top_k_embeds = [all_img_dir_embeds[i] for i in top_k]
        top_k_image_name = [all_image_name[i] for i in top_k]
        
        return {f"Top {k} Sim Images": top_k_image_name, f"Top {k} Sim Image Embeds": top_k_embeds}

if __name__ == "__main__":
    model_name = "ViT-B/32"
    clip_model = CLIP(model_name)
    model, preprocess = clip_model.load_image_model()

    # image_path = "/mnt/e/tinkering/clip_image_exp/image_to_image_search/data/dog_and_human_cricket.jpg"
    # candidate_texts = [
    #     "a human running",
    #     "dog running in backyard",
    #     "cat jumping",
    #     "aliens",
    #     "a human playing cricket with dog"
    # ]

    # print(clip_model.get_text_similarity_based_on_single_inp_image(image_path, candidate_texts, model, preprocess))


    inp_image_path = r"/mnt/e/tinkering/clip_image_exp/image_to_image_search/data/dog_running.jpeg"
    ref_image_dir = r"/mnt/e/tinkering/clip_image_exp/image_to_image_search/ref_img"

    clip_model.get_image_to_image_similarity(inp_image_path, ref_image_dir, model, preprocess)









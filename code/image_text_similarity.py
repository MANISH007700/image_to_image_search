import os 
import numpy as np
from typing import List
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import torch
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

    def get_image_similarity(self, image_path: str, candidates_name: list[str], model, preprocess) -> List:

        # loads and return array info of image
        image = preprocess(Image.open(image_path)).unsqueeze(0).to(self.device)
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
    

if __name__ == "__main__":
    model_name = "ViT-B/32"
    clip_model = CLIP(model_name)
    model, preprocess = clip_model.load_image_model()

    image_path = "/mnt/e/tinkering/clip_image_exp/image_to_image_search/data/dog_and_human_cricket.jpg"
    candidate_texts = [
        "a human running",
        "dog running in backyard",
        "cat jumping",
        "aliens",
        "a human playing cricket with dog"
    ]

    print(clip_model.get_image_similarity(image_path, candidate_texts, model, preprocess))



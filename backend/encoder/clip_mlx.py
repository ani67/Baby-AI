import numpy as np
import torch
from transformers import CLIPModel, CLIPProcessor


class CLIPWrapper:
    """
    Loads CLIP via HuggingFace transformers.
    Returns numpy arrays, converted to torch by callers.
    """

    MODEL_NAME = "openai/clip-vit-base-patch32"

    def __init__(self):
        self._processor = CLIPProcessor.from_pretrained(self.MODEL_NAME)
        self._model = CLIPModel.from_pretrained(self.MODEL_NAME)
        self._model.eval()

    def encode_images(self, images: list) -> np.ndarray:
        """
        Encode a list of PIL Images.
        Returns: numpy array of shape (N, 512).
        """
        inputs = self._processor(images=images, return_tensors="pt")
        with torch.no_grad():
            out = self._model.get_image_features(**inputs)
            # transformers 5.x returns BaseModelOutputWithPooling
            if hasattr(out, "pooler_output"):
                features = out.pooler_output
            else:
                features = out
        return features.cpu().numpy()

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        """
        Encode a list of text strings.
        Returns: numpy array of shape (N, 512).
        """
        inputs = self._processor(text=texts, return_tensors="pt",
                                 padding=True, truncation=True,
                                 max_length=77)
        with torch.no_grad():
            out = self._model.get_text_features(**inputs)
            if hasattr(out, "pooler_output"):
                features = out.pooler_output
            else:
                features = out
        return features.cpu().numpy()

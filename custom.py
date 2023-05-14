import torch.nn as nn
from modules.feature_extraction import VGG_FeatureExtractor
from modules.sequence_modeling import BidirectionalLSTM
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.stages = {'Trans': None, 'Feat': 'VGG',
                       'Seq': 'BiLSTM', 'Pred':'CTC'}
        self.FeatureExtraction = VGG_FeatureExtractor(1, 512)
       
        self.FeatureExtraction_output = 512  # int(imgH/16-1) * 512
        self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((None, 1))  # Transform final (imgH/16-1) -> 1

        """ Sequence modeling"""
        self.SequenceModeling = nn.Sequential(
            BidirectionalLSTM(self.FeatureExtraction_output, 256, 256),
            BidirectionalLSTM(256, 256, 256))
        self.SequenceModeling_output = 256


        """ Prediction """
        self.Prediction = nn.Linear(self.SequenceModeling_output, 37)

    def forward(self, input, text, is_train=True):

        """ Feature extraction stage """
        visual_feature = self.FeatureExtraction(input)
        visual_feature = self.AdaptiveAvgPool(visual_feature.permute(0, 3, 1, 2))  # [b, c, h, w] -> [b, w, c, h]
        visual_feature = visual_feature.squeeze(3)

        """ Sequence modeling stage """
        contextual_feature = self.SequenceModeling(visual_feature)
        
        """ Prediction stage """
        prediction = self.Prediction(contextual_feature.contiguous())
        
        return prediction

class RawDataset(Dataset):
    def __init__(self, root):
        self.image_path_list = root
        self.nSamples = len(self.image_path_list)

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
      img = self.image_path_list[index]
      if isinstance(img, str):
        img = Image.open(img).convert('L')
      elif isinstance(img, np.ndarray):
        img = Image.fromarray(img).convert('L')
      return (img, self.image_path_list[index])
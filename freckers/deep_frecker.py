import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class DeepFrecker:
    def __init__(self, model):
        self.model = model

    def run(self, gameboard):

        action_prob, value = self.inference(gameboard, self.model)

        return action_prob[0], value[0]

    
    def inference(self, input_data, model):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.eval()
        model.to(device)

        with torch.no_grad():
            input_tensor = torch.tensor(input_data.copy()).float()
            input_tensor = input_tensor.to(device)
            if len(input_tensor.shape) == 3:
                input_tensor = input_tensor.unsqueeze(0)
            
            img_output, prob_output = model(input_tensor)
            
            img_output = img_output.cpu().numpy()
            prob_output = prob_output.cpu().numpy()
            return img_output, prob_output


"""
Creative Commons Attribution-NonCommercial ShareAlike 4.0 International License  https://creativecommons.org/licenses/by-nc-sa/4.0/
"""

import torch
from diffusers import DiffusionPipeline


class DiffPipe(DiffusionPipeline):
    def __init__(self, model, scheduler):
        super().__init__()

        self.register_modules(model=model, scheduler=scheduler)

    @torch.no_grad()
    def __call__(self, input_tensor, inputs_2d, inputs_2d_crop, context_features):
        predicted_3d_pos = []
        for t in self.scheduler.timesteps:
            timestep = t[None].repeat(input_tensor.shape[0]).long().to(self.device)

            # 1. predict noise model_output
            model_output = self.model(inputs_2d, inputs_2d_crop, input_tensor, context_features, timestep)

            # 2. predict previous mean of image x_t-1 and add variance depending on eta
            # eta corresponds to η in paper and should be between [0, 1]
            # do x_t -> x_t-1
            input_tensor = self.scheduler.step(model_output, t.item(), input_tensor).prev_sample
            predicted_3d_pos.append(input_tensor)
        predicted_3d_pos = torch.stack(predicted_3d_pos, dim=1)

        return predicted_3d_pos

class DiffPipeGaze(DiffusionPipeline):
    def __init__(self, model, scheduler):
        super().__init__()

        self.register_modules(model=model, scheduler=scheduler)

    @torch.no_grad()
    def __call__(self, input_tensor, inputs_2d, inputs_2d_crop, context_features, emb, center):
        predicted_3d_pos = []
        for t in self.scheduler.timesteps:
            timestep = t[None].repeat(input_tensor.shape[0]).long().to(self.device)

            # 1. predict noise model_output
            model_output = self.model(inputs_2d, inputs_2d_crop, input_tensor, context_features, timestep, emb, center)

            # 2. predict previous mean of image x_t-1 and add variance depending on eta
            # eta corresponds to η in paper and should be between [0, 1]
            # do x_t -> x_t-1
            input_tensor = self.scheduler.step(model_output, t.item(), input_tensor).prev_sample
            predicted_3d_pos.append(input_tensor)
        predicted_3d_pos = torch.stack(predicted_3d_pos, dim=1)

        return predicted_3d_pos
    


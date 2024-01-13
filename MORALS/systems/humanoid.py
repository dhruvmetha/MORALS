import numpy as np 
from MORALS.systems.system import BaseSystem

class Humanoid(BaseSystem):
    def __init__(self,**kwargs):
        self.name = "humanoid"

        # self.state_bounds = NotImplementedError
    
    # def get_true_bounds(self):
    #     return NotImplementedError
    
    # def get_bounds(self):
    #     return NotImplementedError 

    def transform(self, s):
        # new_s = np.concatenate([s[:, 21:34], s[:, 34:40]], axis=-1)
        new_s = s
        return new_s
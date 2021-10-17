import torch
import os
import numpy as np
import random
from dekorde.paths import MODEL_DIR
from dekorde.components.transformer import Transformer

    
def seed_everything(seed):
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if use multi-GPU


def save_checkpoint(state, model_name):
    print('Saving model ...')
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)    
    torch.save(state, os.path.join(MODEL_DIR, model_name))
    
    
def load_model(model, model_name):
    model_path = os.path.join(MODEL_DIR, model_name)
    load_state = torch.load(model_path)

    model.load_state_dict(load_state['state_dict'], strict=True)
    
    print("Loading Model from:", model_path, "...Finished.")
    return model
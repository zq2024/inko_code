import os
import torch

def save(obj, path):
    if type(obj) != "INKO.model.INKO":
        print("Save function only for saving INKO solver!")
        return
    (fpath,_) = os.path.split(path)
    if not os.path.isfile(path):
        os.makedirs(fpath)
    torch.save({"INKO":obj},path)
    print("Koopman solver has been saved successfully!")

def load(path):
    f = torch.load(path)
    return f['INKO']
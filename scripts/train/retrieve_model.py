import torch
from model import LMBaseModel, RGCNConcatModel, HalfGCNModel
# dump a model architecture to a file
model_name = "yarongef/DistilProtBert"
lm_model = LMBaseModel(model_name)
rgcn_model = RGCNConcatModel()
gcn_model = HalfGCNModel(3)
with open("lm_model.txt", "w") as f:
    f.write(str(lm_model))

with open("rgcn_model.txt", "w") as f:
    f.write(str(rgcn_model))

with open("gcn_model.txt", "w") as f:
    f.write(str(gcn_model))
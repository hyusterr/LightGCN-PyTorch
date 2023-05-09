import sys
import torch

model = torch.load(sys.argv[1])
model.eval()
all_users, all_items = model.computer()

    

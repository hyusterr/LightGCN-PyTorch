import world
import utils
from world import cprint
import torch
import numpy as np
import pandas as pd
from tensorboardX import SummaryWriter
import time
import Procedure
from os.path import join
# ==============================
utils.set_seed(world.seed)
print(">>SEED:", world.seed)
# ==============================
import register
from register import dataset
import matplotlib.pyplot as plt

Recmodel = register.MODELS[world.model_name](world.config, dataset)
Recmodel = Recmodel.to(world.device)
bpr = utils.BPRLoss(Recmodel, world.config)

weight_file = utils.getFileName()
print(f"load and save to {weight_file}")
if world.LOAD:
    try:
        Recmodel.load_state_dict(torch.load(weight_file,map_location=torch.device('cpu')))
        world.cprint(f"loaded model weights from {weight_file}")
    except FileNotFoundError:
        print(f"{weight_file} not exists, start from beginning")
Neg_k = 1

Recmodel.eval()
all_users, all_items = Recmodel.computer()
print('user matrix', all_users.size())
print('item matrix', all_items.size())

# user_embdf = pd.DataFrame(all_users.detach().cpu().numpy())
# item_embdf = pd.DataFrame(all_items.detach().cpu().numpy())
user_embdf = all_users.detach().cpu().numpy()[1:]
item_embdf = all_items.detach().cpu().numpy()[1:]


item_columns = "id | title | date | video | IMDb URL | unknown | Action | Adventure | Animation | Children's | Comedy | Crime | Documentary | Drama | Fantasy | Film-Noir | Horror | Musical | Mystery | Romance | Sci-Fi | Thriller | War | Western".split(" | ")
item_info = pd.read_csv('../data/ml-100k/u.item', header=None, sep='|', encoding='latin1', names=item_columns)
item_type = item_info[item_columns[-19:]].idxmax(1)

user_columns = 'user id | age | gender | occupation | zip code'.split(' | ')
user_info = pd.read_csv('../data/ml-100k/u.user', header=None, sep='|', encoding='latin1', names=user_columns)

from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, random_state=0)
item_2d = tsne.fit_transform(item_embdf)
plt.figure(figsize=(10, 10))
plt.scatter(item_2d[:, 0], item_2d[:, 1], label=item_type)
plt.savefig("lightgcn_ml-100k_itememb.png")


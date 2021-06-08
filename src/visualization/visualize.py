from torch.nn.modules.container import Sequential
from src.models.model import MNIST_NET
from src.data.make_dataset import mnist
import torch
import torch.nn as nn
import argparse
import sys
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns

parser = argparse.ArgumentParser(description='Training arguments')
parser.add_argument('--modelpath', default='models/0_checkpoint.pth')
parser.add_argument('--n_components', default=2)
args = parser.parse_args(sys.argv[2:])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
_, testset = mnist()
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

model = MNIST_NET().to(device)
state_dict = torch.load(args.modelpath)
model.load_state_dict(state_dict)
modules = list(model.children())[:-1]
modules2 = list(model.classifier.children())[:-4]
model = nn.Sequential(*modules)
model2 = nn.Sequential(*modules2)

out = []
lab = []
with torch.no_grad():
    # validation pass here
    model.eval()
    for batch_idx, (images, labels) in enumerate(testloader):
        x = model(images)
        x = x.view(x.size(0), -1)
        x = model2(x)
        out.append(x)
        lab.append(labels)
out = torch.cat(out, dim=0)
lab = torch.cat(lab, dim=0)
tsne = TSNE(args.n_components)

x_new = tsne.fit_transform(out)

df = pd.DataFrame(lab, columns=['label'])
df['tsne-1'] = x_new[:, 0]
df['tsne-2'] = x_new[:, 1]

plt.figure(figsize=(16, 10))
sns.scatterplot(x="tsne-1",
                y="tsne-2",
                hue="label",
                palette=sns.color_palette("hls", 10),
                data=df,
                legend="full",
                alpha=0.3)
plt.savefig('reports/figures/tsne_features.png')
from src.models.model import MNIST_NET
from src.data.make_dataset import mnist
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
trainset, _ = mnist()

trainloader = torch.utils.data.DataLoader(trainset,
                                          batch_size=64,
                                          shuffle=True)

model = MNIST_NET().to(device)

output = model(next(iter(trainloader))[0])
assert output.shape[0] == 64 and output.shape[
    1] == 10, "Model output shapes are wrong should be [b,10]"

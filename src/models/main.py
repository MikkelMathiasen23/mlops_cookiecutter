import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter

import wandb
from src.data.make_dataset import mnist
from src.models.model import MNIST_NET
import time

writer = SummaryWriter()


class TrainOREvaluate(object):
    """ Helper class that will help launch class methods as commands
        from a single script
    """
    def __init__(self):
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        parser = argparse.ArgumentParser(
            description="Script for either training or evaluating",
            usage="python main.py <command>")
        parser.add_argument("command", help="Subcommand to run")
        args = parser.parse_args(sys.argv[1:2])
        if not hasattr(self, args.command):
            print('Unrecognized command')

            parser.print_help()
            exit(1)
        # use dispatch pattern to invoke method with same name
        getattr(self, args.command)()

    def train(self):
        print("Training day and night")
        parser = argparse.ArgumentParser(description='Training arguments')
        parser.add_argument('--lr', default=0.003)
        # add any additional argument that you want
        parser.add_argument('--epoch', default=10)
        parser.add_argument('--model_version', default=00)
        parser.add_argument('--batch_size', default=64)

        args = parser.parse_args(sys.argv[2:])
        print(args)
        print('device: ', self.device)
        hyperparameters = {
            'batch_size': args.batch_size,
            'lr': args.lr,
            'epoch': args.epoch
        }
        wandb.init(config=hyperparameters)

        model = MNIST_NET().to(self.device)
        wandb.watch(model, log_freq=10)
        criterion = nn.CrossEntropyLoss()  # criterion = nn.NLLLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        trainset, _ = mnist()
        trainloader = torch.utils.data.DataLoader(trainset,
                                                  batch_size=args.batch_size,
                                                  shuffle=True)
        train_losses = []
        steps = 0
        for e in range(int(args.epoch)):
            print('Epoch: ', e)
            running_loss = 0
            for batch_idx, (images, labels) in enumerate(trainloader):
                model.train()
                if batch_idx == 0:
                    wandb.log({
                        "Image examples":
                        wandb.Image(
                            images.view(args.batch_size, 1, 28, 28)[0,
                                                                    0, :, :])
                    })

                images = images.to(self.device)
                labels = labels.to(self.device)
                steps += 1
                optimizer.zero_grad()

                log_ps = model(images)
                loss = criterion(log_ps, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss

                train_losses.append(loss.item())
                # Log train loss
                writer.add_scalar('Loss/train', loss, batch_idx * (e + 1))
                wandb.log({"loss": loss})
                # Log class probabilities
                writer.add_histogram('Class probabilities', log_ps.flatten(),
                                     batch_idx * (e + 1))
            print("Train loss:", running_loss.item())

        torch.save(model.state_dict(),
                   'models/' + str(args.model_version) + '_checkpoint.pth')

        # Log model graph
        writer.add_graph(model, input_to_model=images)
        writer.add_hparams(hyperparameters, {'steps': steps})
        plt.figure()
        x = np.arange(steps)
        plt.plot(x, train_losses, label='Train loss')
        plt.legend(loc='upper right')
        plt.savefig('reports/figures/train_loss.png')

        return train_losses

    def para(self):
        parser = argparse.ArgumentParser(description='Parallel arguments')
        parser.add_argument('--batch_size', default=64)
        parser.add_argument('--rep', default=1)
        args = parser.parse_args(sys.argv[2:])
        model = nn.DataParallel(MNIST_NET().to(self.device))
        model_no = (MNIST_NET().to(self.device))
        trainset, _ = mnist()

        trainloader = torch.utils.data.DataLoader(trainset,
                                                  batch_size=int(
                                                      args.batch_size),
                                                  shuffle=True)
        batch, _ = next(iter(trainloader))

        start = time.time()
        for r in range(int(args.rep)):
            model(batch)
        end = time.time()
        print('Timing parallel: ', end - start)
        for r in range(int(args.rep)):
            model_no(batch)
        end = time.time()
        print('Timing original: ', end - start)

    def evaluate(self):
        print("Evaluating until hitting the ceiling")
        parser = argparse.ArgumentParser(description='Training arguments')
        parser.add_argument('--load_model_from', default="")
        # add any additional argument that you want
        args = parser.parse_args(sys.argv[2:])
        print(args)

        if args.load_model_from:
            state_dict = torch.load(args.load_model_from)

        model = MNIST_NET().to(self.device)
        model.load_state_dict(state_dict)

        _, testset = mnist()
        testloader = torch.utils.data.DataLoader(testset,
                                                 batch_size=64,
                                                 shuffle=False)

        with torch.no_grad():
            # validation pass here
            accuracy = 0
            model.eval()
            for batch_idx, (images, labels) in enumerate(testloader):

                log_ps = model(images.to(self.device))
                ps = torch.exp(log_ps)
                top_p, top_class = ps.topk(1, dim=1)

                equals = top_class == labels.view(*top_class.shape).to(
                    self.device)
                accuracy += torch.mean(equals.type(torch.FloatTensor))
                writer.add_scalar('Accuracy/test', accuracy, batch_idx)
            accuracy = accuracy / len(testloader)
            print('Accuracy: ', accuracy.item())

            return accuracy


if __name__ == '__main__':
    TrainOREvaluate()

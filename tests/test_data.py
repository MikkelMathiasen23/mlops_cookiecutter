import torch

from src.data.make_dataset import mnist

trainset, testset = mnist()

assert len(trainset) == 60000, 'Training data should be 60000 samples'
assert len(testset) == 10000, 'Test data should be 10000 samples'

assert trainset.data.shape[0] == 60000 and trainset.data.shape[
    1] == 28 and trainset.data.shape[
        2] == 28, "Each datapoint (in train) should have shape [1,28,28]"
assert testset.data.shape[0] == 10000 and testset.data.shape[
    1] == 28 and testset.data.shape[
        2] == 28, "Each datapoint (in test) should have shape [1,28,28]"

assert (trainset.targets.unique() == torch.tensor(
    [0, 1, 2, 3, 4, 5, 6, 7, 8,
     9])).all, "Class labels [0,1,2,3,4,5,6,7,8,9 should be present"

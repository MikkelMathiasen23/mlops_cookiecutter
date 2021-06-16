import torch
import torchvision.models as models

resnet18 = models.resnet18(pretrained=True)
script_model = torch.jit.script(resnet18)
script_model.save('models_torchserve/deployable_model.pt')

if __name__ == '__main__':
    resnet18.eval()
    script_model.eval()
    random_input = torch.rand(1, 3, 224, 224)
    unscripted_top5_indicies = torch.topk(resnet18(random_input), 5)[1]
    scripted_top5_indicies = torch.topk(script_model(random_input), 5)[1]

    print(
        "Top class and indicies for non serialized: {}. \nTop class and indicies for serialzed: {}"
        .format(unscripted_top5_indicies, scripted_top5_indicies))

    assert torch.allclose(unscripted_top5_indicies,
                          scripted_top5_indicies), "Not allclose"

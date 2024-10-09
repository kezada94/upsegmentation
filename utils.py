import argparse

import torch


class FloatAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        if len(values) not in [1, 3]:
            raise argparse.ArgumentTypeError("Must provide either one float or three floats.")
        setattr(namespace, self.dest, values)


def labels_to_one_hot(target):
    return target
    if not torch.is_tensor(target):
        target = torch.tensor(target, dtype=torch.long)
    else:
        target = target.to(torch.long)

    # print(target.shape)
    gre = torch.nn.functional.one_hot(target, -1)
    # print(" GRE: ", gre.shape)
    result = gre.transpose(1, 4).squeeze(-1)
    # print(" RESULT:", result.shape)
    # show an image of the result class 0 and 1
    # import matplotlib.pyplot as plt
    
    # fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    # axes[0].imshow(result[0][0].cpu().numpy(), cmap='gray')
    # axes[0].set_title("Class 0")
    # plt.show()


    # result = torch.nn.functional.one_hot(target, 2).transpose(1, 4)[:,:,:,:,0]
    return result


def one_hot_to_labels(target):
    middle = torch.argmax(target, dim=1)
    result = torch.unsqueeze(middle, dim=1)
    return result

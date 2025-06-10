import numpy as np
import os
import sys
import torch
from method.evaluation_metrics_3d import _pairwise_EMD_CD_
current_file_path = os.path.abspath(__file__)
sys.path.append(os.path.join(os.sep, *current_file_path.split(os.sep)[:current_file_path.split(os.sep).index("weight_memorization") + 1], "visualization_util"))
from plot_scatter_plot import draw, randomly_select_100

if __name__ == "__main__":
    generated = torch.load("./data/gen_point_clouds.pth").cuda()
    training = torch.load("./data/train_point_clouds.pth").cuda()
    test = torch.load("./data/test_point_clouds.pth").cuda()
    noise002 = torch.load("./data/noise0.02_point_clouds.pth").cuda()
    noise004 = torch.load("./data/noise0.04_point_clouds.pth").cuda()

    novelty = []
    performance = []
    for type in [training, generated, noise002, noise004]:
        cd_to_train, _ = _pairwise_EMD_CD_(training, type, batch_size=64)
        if type is training:
            cd_to_train.fill_diagonal_(float('inf'))
        cd_to_train_min = cd_to_train.min(dim=0).values
        cd_to_test, _ = _pairwise_EMD_CD_(test, type, batch_size=64)
        cd_to_test_min = cd_to_test.min(dim=0).values
        cd_to_train_min, cd_to_test_min = randomly_select_100(cd_to_train_min, cd_to_test_min)
        novelty.append(cd_to_train_min.cpu().numpy())
        performance.append(cd_to_test_min.cpu().numpy())

    draw(novelty, performance, "min dist. to training set", "min dist. to test set", ["training", "generated", "noise0.02", "noise0.04"], os.path.abspath("figures/performance_vs_novelty.png"))
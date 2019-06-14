import time
import torch
from dataset import CustomDataset, TestDataSet
import torch.nn.functional as F
import numpy as np
from utils import accuracy

from torch.utils.data import DataLoader


def test(data_loader, txtfile):

    net = torch.load('trained_models/acc_90.pkl')

    theta_c = 0.5
    crop_size = (256, 256)  # size of cropped images for 'See Better'

    # metrics initialization
    batches = 0
    epoch_loss = 0
    epoch_acc = np.array([0, 0, 0], dtype='float')  # top - 1, 3, 5

    net.eval()

    with open(txtfile, 'w') as file:
        with torch.no_grad():
            for i, X in enumerate(data_loader):

                # obtain data
                X = X.to(torch.device("cuda"))

                ##################################
                # Raw Image
                ##################################
                y_pred_raw, feature_matrix, attention_map = net(X)

                ##################################
                # Object Localization and Refinement
                ##################################
                # crop_mask = F.upsample_bilinear(attention_map, size=(X.size(2), X.size(3))) > theta_c
                crop_mask = F.interpolate(attention_map, size=(X.size(2), X.size(3))) > theta_c
                crop_images = []
                for batch_index in range(crop_mask.size(0)):
                    nonzero_indices = torch.nonzero(crop_mask[batch_index, 0, ...])
                    height_min = nonzero_indices[:, 0].min()
                    height_max = nonzero_indices[:, 0].max()
                    width_min = nonzero_indices[:, 1].min()
                    width_max = nonzero_indices[:, 1].max()
                    crop_images.append(F.upsample_bilinear(
                        X[batch_index:batch_index + 1, :, height_min:height_max, width_min:width_max], size=crop_size))
                crop_images = torch.cat(crop_images, dim=0)

                y_pred_crop, _, _ = net(crop_images)

                # final prediction
                y_pred = (y_pred_raw + y_pred_crop) / 2

                for y in y_pred:
                    _, pred_idx = y.topk(1, 0, True, True)
                    file.write(str(pred_idx.item()+1) + ' ')

                    s = torch.nn.Softmax(dim=0)
                    confidence_score = max(s(y)).item()
                    file.write(str(confidence_score) + '\n')


if __name__ == '__main__':

    testset = TestDataSet('./Dataset', cropped=True)

    test_loader = DataLoader(testset, 64, shuffle=False)
    test(test_loader, 'predict.txt')

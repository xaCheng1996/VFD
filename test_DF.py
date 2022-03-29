from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from tqdm import tqdm
import numpy as np
import torch


def auc(real, fake):
    label_all = []
    target_all = []
    for ind in real:
        target_all.append(1)
        label_all.append(-ind)
    for ind in fake:
        target_all.append(0)
        label_all.append(-ind)

    from sklearn.metrics import roc_auc_score
    return roc_auc_score(target_all, label_all)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == '__main__':
    opt = TestOptions().parse()
    opt.num_threads = 4
    opt.batch_size = 1
    opt.serial_batches = False
    opt.no_flip = True
    opt.display_id = -1
    opt.mode = 'test'
    dataset = create_dataset(opt)
    model = create_model(opt)
    model.setup(opt)

    model.eval()

    dataset_size = len(dataset)
    print('The number of test images dir = %d' % dataset_size)

    total_iters = 0
    label = None
    real = []
    fake = []

    with tqdm(total=dataset_size) as pbar:
        for i, data in enumerate(dataset):
            input_data = {'img_real': data['img_real'],
                          'img_fake': data['img_fake'],
                          'aud_real': data['aud_real'],
                          'aud_fake': data['aud_fake'],
                          }
            model.set_input(input_data)

            dist_AV, dist_VA = model.val()
            real.append(dist_AV.item())
            for i in dist_VA:
                fake.append(i.item())
            total_iters += 1
            pbar.update()

    print('The auc is %.3f'%(auc(real, fake)))

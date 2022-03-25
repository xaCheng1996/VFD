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
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 4  # test code only supports num_threads = 1
    opt.batch_size = 1  # test code only supports batch_size = 1
    opt.serial_batches = False  #r disable data shuffling; comment this line if results on randomly chosen images ae needed.
    opt.no_flip = True  # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1  # no visdom display; the test code saves the results to a HTML file.
    opt.mode = 'test'
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)  # create a model given opt.model and other options
    model.setup(opt)  # regular setup: load and print networks; create schedulers

    if opt.eval:
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

    import matplotlib.pyplot as plt
    min_val = min(min(fake), min(real))
    fake_new = []
    real_new = []
    for i in real:
        real_new.append((i-min_val))
        # real_new.append((i - min_val))
    for i in fake:
        fake_new.append((i-min_val))
        # fake_new.append((i - min_val))
    real = real_new
    fake = fake_new
    legend_font = {"family": "Times New Roman",
                   "size": 18,
                   }
    plt.xticks([], fontproperties='Times New Roman', fontsize=20)
    plt.yticks(np.arange(0, 800, 100), fontproperties='Times New Roman', fontsize=20)
    plt.hist(fake, bins=20, facecolor="pink", edgecolor="black", alpha=0.7, label='Fake')
    plt.hist(real, bins=20, facecolor="yellowgreen", edgecolor="black", alpha=0.7, label='Real')
    plt.legend(loc='upper left', framealpha=1, prop=legend_font)
    # plt.legend()
    plt.show()

    print('The auc is %.3f'%(auc(real, fake)))

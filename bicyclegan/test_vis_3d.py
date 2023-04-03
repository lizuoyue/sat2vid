import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_gif
from itertools import islice
from util import util
from util import html
import torch
from gpu import get_unused_gpu

if __name__ == '__main__':
    # options
    opt = TestOptions().parse()
    opt.num_threads = 1   # test code only supports num_threads=1
    opt.batch_size = 1   # test code only supports batch_size=1
    opt.serial_batches = True  # no shuffle

    opt.max_len_path = 15
    opt.sync = True
    opt.rotate_data = False

    # create dataset
    # torch.manual_seed(1994)
    dataset = create_dataset(opt)
    print(len(dataset))

    model = create_model(opt)
    model.setup(opt)
    # model.eval()
    print(f'Loading model {opt.model} in {model.device}')

    # create website
    web_dir = os.path.join(opt.results_dir, ((opt.phase+'_sync') if opt.sync else opt.phase)) \
        + ('_per_frame' if opt.per_frame_test else '_all_frames') + f'_{opt.num_frames}frames'
    webpage = html.HTML(web_dir, 'Training = %s, Phase = %s, Class =%s' % (opt.name, opt.phase, opt.name))

    # sample random z
    if opt.sync:
        if not opt.local_encoder:
            z_samples = model.get_z_random(opt.n_samples + 1, opt.nz)
        else:
            z_samples = model.get_z_random_local([opt.n_samples + 1, 8, opt.nz])

    # test stage
    for i, data in enumerate(islice(dataset, opt.num_test)):

        if i != 86:
            continue

        model.set_input_point_cloud(data)
        print('process input image %3.3d/%3.3d' % (i, opt.num_test))

        if not opt.sync:
            if not opt.local_encoder:
                z_samples = model.get_z_random(opt.n_samples + 1, opt.nz)
            else:
                z_samples = model.get_z_random_local([opt.n_samples + 1, 8, opt.nz])
        for nn in range(opt.n_samples + 1):
            encode = nn == 0 and not opt.no_encode
            if opt.dataset_mode.endswith('pc'):
                real_A, fake_B, real_B = model.test_pc(z_samples[[nn]], encode=encode)
            else:
                real_A, fake_B, real_B = model.test(z_samples[[nn]], encode=encode, per_frame=opt.per_frame_test)
                real_A = real_A[:,:,:3]
            # for item in [real_A, fake_B, real_B]:
            #     print(item.shape, item.min(), item.max())
            # input()
            if nn == 0:
                images = [real_A, real_B, fake_B]
                names = ['input', 'ground truth', 'encoded']
            else:
                images.append(fake_B)
                names.append('random_sample%2.2d' % nn)

        img_path = 'input_%3.3d' % i
        save_gif(webpage, images, names, img_path, num_frames=opt.num_frames, aspect_ratio=opt.aspect_ratio, width=opt.crop_size, save_img=opt.save_img)

    webpage.save()

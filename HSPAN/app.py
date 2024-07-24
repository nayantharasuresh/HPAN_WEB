from flask import Flask, render_template, request, redirect, url_for, send_file
import os
import PIL
from PIL import Image
import torch
import argparse
import os
import src.utility as utility
import src.data as data
# import src.model as model
import src.loss as loss
from src.trainer import Trainer
import argparse
import src.template as template
from importlib import import_module
# from dataloader import MSDataLoader
from torch.utils.data import dataloader
from torch.utils.data import ConcatDataset
import torch.nn as nn
print(os.listdir())
UPLOAD_FOLDER = 'HSPAN/static'
app = Flask(__name__, template_folder='template', static_folder='static')

config_args = dict(
    debug=True,
    template='.',
    n_threads=2,
    cpu=True,
    n_GPUs=1,
    seed=1,
    local_rank=0,
    dir_data='.',
    dir_demo='Demo',
    data_train='DIV2K',
    data_test='Demo',
    data_range='1-800/1-5',
    ext='sep',
    scale='4',
    patch_size=192,
    rgb_range=1,
    hidden_nums=128,
    n_colors=3,
    chunk_size=144,
    n_hashes=4,
    n_margin=8,
    chop=True,
    no_augment=False,
    model='HSPAN',
    act='relu',
    pre_train='experiment/HSPAN_x4/model/HSPAN_x4.pt',
    extend='.',
    n_resblocks=4,
    n_reslayers=20,
    norm_group_num=16,
    head_size=32,
    n_feats=192,
    res_scale=0.1,
    shift_mean=True,
    softmax_flag=True,
    orthogonal_flag=True,
    relu_flag=0,
    orthogonal_vec_num=0,
    dilation=False,
    precision='single',
    G0=64,
    RDNkSize=3,
    RDNconfig='B',
    depth=12,
    n_resgroups=10,
    reduction=2,
    reset=False,
    test_every=1000,
    topk=128,
    epochs=1000,
    batch_size=16,
    split_batch=1,
    self_ensemble=False,
    test_only=True,
    gan_k=1,
    lr=0.0001,
    low_threshold=0.15,
    sigma=1.0,
    decay='200',
    gamma=0.5,
    optimizer='ADAM',
    momentum=0.9,
    betas=(0.9, 0.999),
    epsilon=1e-08,
    weight_decay=0,
    gclip=0,
    loss='1*L1',
    skip_threshold=100000000.0,
    save='HSPAN_x4_results',
    load='',
    resume=0,
    save_models=True,
    print_every=100,
    save_results=True,
    save_gt=False)

args = argparse.Namespace(**config_args)

template.set_template(args)

args.scale = list(map(lambda x: int(x), args.scale.split('+')))
args.data_train = args.data_train.split('+')
args.data_test = args.data_test.split('+')

if args.epochs == 0:
    args.epochs = 1e8

for arg in vars(args):
    if vars(args)[arg] == 'True':
        vars(args)[arg] = True
    elif vars(args)[arg] == 'False':
        vars(args)[arg] = False

torch.manual_seed(args.seed)
checkpoint = utility.checkpoint(args)


def get_image_resolution(image):
    img = PIL.Image.open(image)
    wid, hgt = img.size
    return str(wid) + "x" + str(hgt)


from PIL import Image


class MyConcatDataset(ConcatDataset):
    def __init__(self, datasets):
        super(MyConcatDataset, self).__init__(datasets)
        self.train = datasets[0].train

    def set_scale(self, idx_scale):
        for d in self.datasets:
            if hasattr(d, 'set_scale'): d.set_scale(idx_scale)


class Data:
    def __init__(self, args):
        self.loader_train = None
        if not args.test_only:
            datasets = []
            for d in args.data_train:
                module_name = d if d.find('DIV2K-Q') < 0 else 'DIV2KJPEG'
                m = import_module('data.' + module_name.lower())
                datasets.append(getattr(m, module_name)(args, name=d))

            self.loader_train = dataloader.DataLoader(
                MyConcatDataset(datasets),
                batch_size=args.batch_size,
                shuffle=True,
                pin_memory=not args.cpu,
                num_workers=args.n_threads,
            )

        self.loader_test = []
        for d in args.data_test:
            if d in ['Set5', 'Set14', 'B100', 'Urban100', 'Manga109']:
                m = import_module('data.benchmark')
                testset = getattr(m, 'Benchmark')(args, train=False, name=d)
            else:
                module_name = d if d.find('DIV2K-Q') < 0 else 'DIV2KJPEG'
                m = import_module('src.' + module_name.lower())
                testset = getattr(m, module_name)(args, train=False, name=d)

            self.loader_test.append(
                dataloader.DataLoader(
                    testset,
                    batch_size=1,
                    shuffle=False,
                    pin_memory=not args.cpu,
                    num_workers=args.n_threads,
                )
            )


class Model(nn.Module):
    def __init__(self, args, ckp):
        super(Model, self).__init__()
        print('Making model...')

        self.scale = args.scale
        self.idx_scale = 0
        self.self_ensemble = args.self_ensemble
        self.chop = args.chop
        self.precision = args.precision
        self.cpu = args.cpu
        self.device = torch.device('cpu' if args.cpu else 'cuda')
        self.n_GPUs = args.n_GPUs
        self.save_models = args.save_models

        module = import_module(args.model.lower())
        self.model = module.make_model(args).to(self.device)
        if args.precision == 'half': self.model.half()

        if not args.cpu and args.n_GPUs > 1:
            self.model = nn.DataParallel(self.model, range(args.n_GPUs))

        self.load(
            ckp.dir,
            pre_train=args.pre_train,
            resume=args.resume,
            cpu=args.cpu
        )
        print(self.model, file=ckp.log_file)

    def forward(self, x, idx_scale):
        self.idx_scale = idx_scale
        target = self.get_model()
        if hasattr(target, 'set_scale'):
            target.set_scale(idx_scale)

        if self.self_ensemble and not self.training:
            if self.chop:
                forward_function = self.forward_chop
            else:
                forward_function = self.model.forward

            return self.forward_x8(x, forward_function)
        elif self.chop and not self.training:
            return self.forward_chop(x)
        else:
            return self.model(x)

    def get_model(self):
        if self.n_GPUs == 1:
            return self.model
        else:
            return self.model.module

    def state_dict(self, **kwargs):
        target = self.get_model()
        return target.state_dict(**kwargs)

    def save(self, apath, epoch, is_best=False):
        target = self.get_model()
        torch.save(
            target.state_dict(),
            os.path.join(apath, 'model_latest.pt')
        )
        if is_best:
            torch.save(
                target.state_dict(),
                os.path.join(apath, 'model_best.pt')
            )

        if self.save_models:
            torch.save(
                target.state_dict(),
                os.path.join(apath, 'model_{}.pt'.format(epoch))
            )

    def load(self, apath, pre_train='.', resume=-1, cpu=False):
        if cpu:
            kwargs = {'map_location': lambda storage, loc: storage}
        else:
            kwargs = {}

        if resume == -1:
            self.get_model().load_state_dict(
                torch.load(
                    os.path.join(apath, 'model_latest.pt'),
                    **kwargs
                ),
                strict=False
            )
        elif resume == 0:
            if pre_train != '.':
                print('Loading model from {}'.format(pre_train))
                self.get_model().load_state_dict(
                    torch.load(pre_train, **kwargs),
                    strict=False
                )
        else:
            self.get_model().load_state_dict(
                torch.load(
                    os.path.join(apath, 'model', 'model_{}.pt'.format(resume)),
                    **kwargs
                ),
                strict=False
            )

    def forward_chop(self, x, shave=10, min_size=120000):
        scale = self.scale[self.idx_scale]
        n_GPUs = min(self.n_GPUs, 4)
        b, c, h, w = x.size()
        h_half, w_half = h // 2, w // 2
        h_size, w_size = h_half + shave, w_half + shave
        h_size += 4 - h_size % 4
        w_size += 8 - w_size % 8

        lr_list = [
            x[:, :, 0:h_size, 0:w_size],
            x[:, :, 0:h_size, (w - w_size):w],
            x[:, :, (h - h_size):h, 0:w_size],
            x[:, :, (h - h_size):h, (w - w_size):w]]

        if w_size * h_size < min_size:
            sr_list = []
            for i in range(0, 4, n_GPUs):
                lr_batch = torch.cat(lr_list[i:(i + n_GPUs)], dim=0)
                sr_batch = self.model(lr_batch)
                sr_list.extend(sr_batch.chunk(n_GPUs, dim=0))
        else:
            sr_list = [
                self.forward_chop(patch, shave=shave, min_size=min_size) \
                for patch in lr_list
            ]

        h, w = scale * h, scale * w
        h_half, w_half = scale * h_half, scale * w_half
        h_size, w_size = scale * h_size, scale * w_size
        shave *= scale

        output = x.new(b, c, h, w)
        output[:, :, 0:h_half, 0:w_half] \
            = sr_list[0][:, :, 0:h_half, 0:w_half]
        output[:, :, 0:h_half, w_half:w] \
            = sr_list[1][:, :, 0:h_half, (w_size - w + w_half):w_size]
        output[:, :, h_half:h, 0:w_half] \
            = sr_list[2][:, :, (h_size - h + h_half):h_size, 0:w_half]
        output[:, :, h_half:h, w_half:w] \
            = sr_list[3][:, :, (h_size - h + h_half):h_size, (w_size - w + w_half):w_size]

        return output

    def forward_x8(self, x, forward_function):
        def _transform(v, op):
            if self.precision != 'single': v = v.float()

            v2np = v.data.cpu().numpy()
            if op == 'v':
                tfnp = v2np[:, :, :, ::-1].copy()
            elif op == 'h':
                tfnp = v2np[:, :, ::-1, :].copy()
            elif op == 't':
                tfnp = v2np.transpose((0, 1, 3, 2)).copy()

            ret = torch.Tensor(tfnp).to(self.device)
            if self.precision == 'half': ret = ret.half()

            return ret

        lr_list = [x]
        for tf in 'v', 'h', 't':
            lr_list.extend([_transform(t, tf) for t in lr_list])

        sr_list = [forward_function(aug) for aug in lr_list]
        for i in range(len(sr_list)):
            if i > 3:
                sr_list[i] = _transform(sr_list[i], 't')
            if i % 4 > 1:
                sr_list[i] = _transform(sr_list[i], 'h')
            if (i % 4) % 2 == 1:
                sr_list[i] = _transform(sr_list[i], 'v')

        output_cat = torch.cat(sr_list, dim=0)
        output = output_cat.mean(dim=0, keepdim=True)

        return output


def resize_image(input_image_path, output_image_path, size):
    with Image.open(input_image_path) as img:
        img.thumbnail(size)
        img.save(output_image_path)


@app.route('/')
def index():
    files = os.listdir(UPLOAD_FOLDER)
    image_files = []
    for f in files:
        if f.endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(UPLOAD_FOLDER, f)
            image_size = int(os.path.getsize(image_path)) / (1024)
            image_resolution = get_image_resolution(image_path)
            image_files.append({
                'image': f,
                'size': image_size,
                'resulsion': image_resolution
            })

    return render_template('upload.html', image_files=image_files)


def clear_all_files(directory_path):
    import shutil
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')

    print(f"All files in {directory_path} have been removed.")


@app.route('/download/<filename>')
def download_image(filename):
    clear_all_files('Demo/')
    import shutil
    source_path = f'./HSPAN/static/{filename}'
    destination_path = f'Demo/{filename}'
    sr_filename = f'{filename.split(".")[0]}_x4_SR.png'
    sr_source = f'experiment/HSPAN_x4_results/results-Demo/{sr_filename}'
    shutil.copy(source_path, destination_path)
    print(f"File copy from {source_path} to {destination_path}")

    if checkpoint.ok:
        loader = Data(args)
        _model = Model(args, checkpoint)
        print('Total params: %.2fM' % (sum(p.numel() for p in _model.parameters()) / 1000000.0))
        _loss = loss.Loss(args, checkpoint) if not args.test_only else None
        t = Trainer(args, loader, _model, _loss, checkpoint)
        while not t.terminate():
            t.train()
            t.test()

    checkpoint.done()

    sr_dst = os.path.join(f'{UPLOAD_FOLDER}/SR', sr_filename)
    shutil.copy(sr_source, sr_dst)
    print(f"File copy from {sr_source} to {sr_dst}")
    file_path = os.path.join(f'{UPLOAD_FOLDER}/SR', sr_filename)

    if os.path.exists(file_path):
        return send_file('static/SR/' + sr_filename, as_attachment=True)
    return 'File not found'


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'image' not in request.files:
        return redirect(request.url)
    file = request.files['image']
    if file.filename == '':
        return redirect(request.url)
    if file:
        filename = file.filename
        file.save(os.path.join(UPLOAD_FOLDER, filename))
        input_image_path = os.path.join(UPLOAD_FOLDER, filename)
        output_image_path = os.path.join(UPLOAD_FOLDER, filename)
        size = (128, 128)
        resize_image(input_image_path, output_image_path, size)
        return redirect('/')


if __name__ == '__main__':
    app.run(debug=True)

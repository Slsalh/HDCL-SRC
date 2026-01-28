from transformers import AutoTokenizer
from fairseq.data import data_utils
import torch
from tqdm import tqdm
import argparse
import os
from model.eval import evaluate

from model import MInterface
from data import DInterface
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

import utils
import pandas as pd
import numpy as np
import json
from model.text_attention import generate
from utils import get_hierarchy_info, save_results
import pickle
from collections import defaultdict
torch.set_float32_matmul_precision("high")

class Saver:
    def __init__(self, model, optimizer, scheduler, args):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.args = args

    def __call__(self, score, best_score, name):
        torch.save({'param': self.model.state_dict(),
                    'optim': self.optimizer.state_dict(),
                    'sche': self.scheduler.state_dict() if self.scheduler is not None else None,
                    'score': score, 'args': self.args,
                    'best_score': best_score},
                   name)


parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=3e-5, help='Learning rate.')
parser.add_argument('--data', type=str, default='rcv1', choices=['rcv1', 'bgc', 'aapd'], help='Dataset.')
parser.add_argument('--label_cpt', type=str, default='data/rcv1/rcv1.taxonomy', help='Label hierarchy file.')
parser.add_argument('--batch', type=int, default=12, help='Batch size.')
parser.add_argument('--early-stop', type=int, default=6, help='Epoch before early stop.')
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--name', type=str, required=True, help='A name for different runs.')
parser.add_argument('--update', type=int, default=1, help='Gradient accumulate steps')
parser.add_argument('--warmup', default=2000, type=int, help='Warmup steps.')
parser.add_argument('--contrast', default=1, type=int, help='Whether use contrastive model.')
parser.add_argument('--contrast_mode', default='attentive', type=str, choices=['label_aware', 'fusion', 'attentive', 'simple_contrastive', 'straight_through'], help='Contrastive model type.')
parser.add_argument('--graph', default=1, type=int, help='Whether use graph encoder.')
parser.add_argument('--layer', default=1, type=int, help='Layer of Graphormer.')
parser.add_argument('--multi', default=True, action='store_false', help='Whether the task is multi-label classification.')
parser.add_argument('--lamb', default=1, type=float, help='lambda')
parser.add_argument('--thre', default=0.02, type=float, help='Threshold for keeping tokens. Denote as gamma in the paper.')
parser.add_argument('--tau', default=1, type=float, help='Temperature for contrastive model.')
parser.add_argument('--seed', default=3, type=int, help='Random seed.')
parser.add_argument('--wandb', default=False, action='store_true', help='Use wandb for logging.')
parser.add_argument('--tf_board', default=False, action='store_true', help='Use tensorboard for logging.')
parser.add_argument('--eval_step', default=1000, type=int, help='Evaluation step.')
parser.add_argument('--head', default=4, type=int, help='Number of heads.')
parser.add_argument('--max_epoch', default=100, type=int, help='Maximum epoch.')
parser.add_argument('--wandb_name', default='HDCL-SRC', type=str, help='Wandb project name.')
parser.add_argument('--checkpoint', default=None, type=str, help='Checkpoint path.')
parser.add_argument('--accelerator', default='ddp', type=str, help='Accelerator for training.')
parser.add_argument('--gpus', default='0', type=str, help='GPU for training.')
parser.add_argument('--test_only', default=False, action='store_true', help='Test only mode.')
parser.add_argument('--test_checkpoint', default=None, type=str, help='Test checkpoint path.')
parser.add_argument('--accumulate_step', default=1, type=int, help='Gradient accumulate step.')
parser.add_argument('--decay_epochs', default=0, type=int, help='Decay epochs.')
parser.add_argument('--softmax_entropy', default=False, action='store_true', help='Use softmax+entropy loss.')
parser.add_argument('--ignore_contrastive', default=False, action='store_true', help='Ignore contrastive loss.')
parser.add_argument('--hiera', default=None, type=str, help='Parent label to child label mapping.')
parser.add_argument('--ftm_weight', default=1, type=float, help='Weight for FTM Loss.')
parser.add_argument('--r_hiera', default=None, type=str, help='Child label to parent label mapping.')
parser.add_argument('--tt_weight', default=0.5, type=float, help='Weight for TT Loss.')
parser.add_argument('--depths', default=None, type=float, help='Depth for every label.')


def get_root(path_dict, n):
    ret = []
    while path_dict[n] != n:
        ret.append(n)
        n = path_dict[n]
    ret.append(n)
    return ret


if __name__ == '__main__':
    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        import sys
        sys.exit(1)

    args.do_weighted_label_contrastive = True
    args.skip_batch_sampling = False
    args.hamming_dist_mode = 'depth_weight'

    device = args.device
    print(args)
    loggers = []
    wandb_logger = None
    if args.wandb:
        wandb_logger = WandbLogger(name=args.wandb_name, project='supContrastiveHMTC')
        loggers.append(wandb_logger)

    # log the args to wandb
    if args.wandb:
        wandb_logger.log_hyperparams(args)
    utils.seed_torch(args.seed)
    pl.seed_everything(args.seed)
        
    args.name = args.data + '-' + args.name
    tokenizer = AutoTokenizer.from_pretrained("./pre_trained_model/bert-base-uncased")
    data_path = os.path.join('data', args.data)
    # This load the pertrained bert model
    # the following code needs to have label_dict, num_class, hiera, r_hiera, label_depth, new_label_dict
    if args.data == 'rcv1':
        hiera, _label_dict, r_hiera, label_depth = get_hierarchy_info(os.path.join(data_path, 'rcv1.taxonomy'))
        with open(os.path.join(data_path, 'new_label_dict.pkl'), 'rb') as f:
            label_dict = pickle.load(f)

        new_label_dict = label_dict

        r_hiera = {new_label_dict[_label_dict[k]]: v if (v == 'Root') else new_label_dict[_label_dict[v]] for k, v in r_hiera.items()}

        # {label_name: label_id}
        label_dict = {v: k for k, v in _label_dict.items()}
        num_class = len(label_dict)

        # new_label_dict = {k: rcv_label_amp[v] for k, v in label_dict.items()}
        depths = [label_depth[name] for id, name in label_dict.items()]

    elif args.data == 'bgc':
        hiera, _label_dict, r_hiera, label_depth = get_hierarchy_info(os.path.join(data_path, 'bgc.taxonomy'))
        label_dict = {v: k for k, v in _label_dict.items()}
        new_label_dict = label_dict
        num_class = len(label_dict)
        depths = list(label_depth.values())
    elif args.data == 'aapd':
        hiera, _label_dict, r_hiera, label_depth = get_hierarchy_info(os.path.join(data_path, 'aapd.taxonomy'))
        with open(os.path.join(data_path, 'new_label_dict.pkl'), 'rb') as f:
            label_dict = pickle.load(f)
        label_dict = {v: k for k, v in label_dict.items()}
        new_label_dict = label_dict
        num_class = len(label_dict)
        depths = [label_depth[name] for id, name in label_dict.items()]

    args.hiera = hiera
    args.r_hiera = r_hiera

    if not os.path.exists(os.path.join('checkpoints', args.name)):
        os.makedirs(os.path.join('checkpoints', args.name))
    # store the label_dict as a json file
    with open(os.path.join('checkpoints', args.name, 'label_dict.json'), 'w') as f:
        json.dump(label_dict, f)

    def get_path(label):
        path = []
        
        if args.data == 'rcv1':
            _, _, rcv_r_hiera, _ = get_hierarchy_info(os.path.join(data_path, 'rcv1.taxonomy'))
            while label != 'Root':
                path.insert(0, label)
                label = rcv_r_hiera[label]
        else:
            while label != 'Root':
                path.insert(0, label)
                label = r_hiera[label]
        return path  # 包含从根节点到当前标签的所有标签

    if ('rcv' in data_path):
        print(_label_dict)
        label_path = {k: get_path(k) for k, v in _label_dict.items()}
    elif ('bgc' in data_path):
        label_path = {k: get_path(k) for k, v in _label_dict.items()}
    else:
        label_path = {k: get_path(k) for k, v in label_dict.items()}

    args.depths = depths
    args.label_path = label_path
    args._label_dict = label_dict  # {label_id: label_name}

    if args.test_only:
        print("start test-=======================================")
        if args.test_checkpoint is None:
            raise ValueError('Please specify the checkpoint path for testing.')

        checkpoint_model_path = os.path.join('checkpoints', args.test_checkpoint)
        args.save_path = checkpoint_model_path + '_save/'
        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path)

        if os.path.exists(os.path.join(args.save_path, 'attns.pkl')):
            # delete the attention files
            os.remove(os.path.join(args.save_path, 'attns.pkl'))
        if os.path.exists(os.path.join(args.save_path, 'indices.pkl')):
            # delete the indices files
            pass
        if os.path.exists(os.path.join(args.save_path, 'labels.pkl')):
            # delete the labels files
            os.remove(os.path.join(args.save_path, 'labels.pkl'))
        if os.path.exists(os.path.join(args.save_path, 'input_ids.pkl')):
            # delete the input_ids files
            os.remove(os.path.join(args.save_path, 'input_ids.pkl'))

        data_module = DInterface(args=args, tokenizer=tokenizer, label_depths=depths, device=device, data_path=data_path, label_dict=label_dict)
        model = MInterface.load_from_checkpoint(checkpoint_model_path, args=args, num_labels=num_class, label_depths=depths, device=device, data_path=data_path, label_dict=label_dict,
                                                new_label_dict=new_label_dict, r_hiera=r_hiera)

        trainer = Trainer(accelerator=args.accelerator, strategy='auto')
        trainer.test(model, datamodule=data_module)

        import sys
        sys.exit(1)

    data_module = DInterface(args=args, tokenizer=tokenizer, label_depths=depths, device=device, data_path=data_path, label_dict=label_dict)

    print("label_dict:", label_dict)
    print("r_hiera:", r_hiera)

    if args.test_checkpoint is None:
        model = MInterface(args, num_labels=num_class, label_depths=depths, device=device, data_path=data_path, label_dict=label_dict, new_label_dict=new_label_dict, r_hiera=r_hiera)
        args.save_path = os.path.join('checkpoints', args.name + '_save/')
        
    else:
        checkpoint_model_path = os.path.join('checkpoints', args.test_checkpoint)
        args.save_path = checkpoint_model_path + '_save/'
        model = MInterface.load_from_checkpoint(checkpoint_model_path, args=args, num_labels=num_class, label_depths=depths, device=device, data_path=data_path, label_dict=label_dict,
                                                new_label_dict=new_label_dict, r_hiera=r_hiera)
    
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
        
    if args.wandb:
        wandb_logger.watch(model)

    checkpoint_callback = ModelCheckpoint(monitor='val/macro_f1', mode='max', save_top_k=1,
                                          dirpath=os.path.join('checkpoints', args.name),
                                          filename=args.name + '-{epoch}', save_on_train_epoch_end=False)
    trainer = Trainer(max_epochs=args.max_epoch, strategy="auto",
                      accelerator=args.accelerator, logger=wandb_logger if args.wandb else None,
                      accumulate_grad_batches=args.accumulate_step,
                      default_root_dir=os.path.join('checkpoints', args.name),
                      gradient_clip_val=1.0, callbacks=[
            EarlyStopping(monitor='val/macro_f1', patience=15, mode='max', check_on_train_epoch_end=False),
            checkpoint_callback]
                      )
    trainer.fit(model, data_module)

    trainer.test(
        datamodule=data_module,
        ckpt_path='best'
    )

    import sys
    sys.exit(1)


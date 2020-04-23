import os
import torch
import logging
import random
import numpy as np
import json
from argparse import ArgumentParser

from src import lightning
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger


def set_global_seed(seed: int):
    """
    Reason from pytorch-lightning maintainer William Falcon:
    "setting the seed inside lightning seems like it hides too much according to ppl I’ve discussed with.
    i don’t want to have people wondering if lightning is doing anything with seeds.
    setting seeds and such is deferred to the user."
    proof: https://github.com/PyTorchLightning/pytorch-lightning/issues/37
    """

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_args():

    parser = ArgumentParser(add_help=False)

    parser.add_argument('--model_type', type=str, default='seq2seq')
    parser.add_argument('--data_source', type=str, default='opensubtitles')
    parser.add_argument('--data_dir', type=str, default='./data/opensubtitles')
    parser.add_argument('--checkpoint_path', type=str, default='./data/opensubtitles/checkpoint')
    parser.add_argument('--project_name', type=str, default='LightningConversation')
    parser.add_argument('--max_norm', type=float, default=2.5)
    parser.add_argument('--distributed_backend', type=str, default='ddp')
    parser.add_argument('--gpus', type=int, default=1 if torch.cuda.is_available() else 0)
    parser.add_argument('--n_grad_accumulate', type=int, default=1)
    parser.add_argument('--batching_type', type=str, default='db')
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--max_length', type=int, default=64)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--seq2seq_prob', type=int, default=0.5)

    parser = lightning.LightningConversation.add_model_specific_args(parser)

    parsed_args = parser.parse_args()

    return parsed_args


if __name__ == '__main__':

    logger = logging.getLogger(__file__)

    args = get_args()

    set_global_seed(args.seed)

    logging.basicConfig(level=logging.INFO)

    if args.model_type in ['dialo_gpt', 'gpt']:
        model = lightning.LightningDialogGPT(args)
    else:
        model = lightning.LightningDialoUnifiedTransformer(args)

    try:
        with open('comet_api_key.txt') as f:
            comet_api_key = f.read().strip()
    except FileNotFoundError:
        comet_api_key = None

    # comet logger don't work with comet logger
    if comet_api_key is not None and args.distributed_backend == 'dp':
        from pytorch_lightning.loggers import CometLogger
        logger = CometLogger(api_key=comet_api_key,
                             project_name=args.project_name)
        logger.experiment.log_parameters(args.__dict__)
        logging.info('Use CometML Logger')
    else:
        logger = TensorBoardLogger(save_dir=os.path.join(os.getcwd(), args.data_dir),
                                   name=args.project_name)
        logging.info('Use TensorBoard Logger')

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        filepath=os.path.join(os.getcwd(), args.checkpoint_path),
        save_top_k=1,
        verbose=True,
        monitor='val_loss',
        mode='min',
        prefix=args.model_type
    )

    try:
        import apex
        use_amp = True
        precision = 16
    except ModuleNotFoundError:
        use_amp = False
        precision = 32
        logging.info('Train without amp, you can install it with command: make install-apex')

    with open(os.path.join(args.data_dir, 'hparams.json'), mode='w') as file_object:
        json.dump(args.__dict__, file_object)

    trainer = pl.Trainer(logger=logger,
                         accumulate_grad_batches=args.n_grad_accumulate,
                         use_amp=use_amp,
                         precision=precision,
                         gradient_clip=args.max_norm,
                         distributed_backend=args.distributed_backend,
                         gpus=args.gpus,
                         val_check_interval=5000,
                         num_sanity_val_steps=0,
                         log_save_interval=10,
                         progress_bar_refresh_rate=10,
                         checkpoint_callback=checkpoint_callback)

    trainer.fit(model)

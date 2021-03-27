import os
import argparse
#from solver_cumulant import Solver
from solver import Solver
from data_loader import get_loader, TestDataset
from torch.backends import cudnn
import config as cfg


def str2bool(v):
    return v.lower() in ('true')

def main(config):
    # For fast training.
    cudnn.benchmark = True

    # Create directories if not exist.
    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)
    if not os.path.exists(config.model_save_dir):
        os.makedirs(config.model_save_dir)
    if not os.path.exists(config.sample_dir):
        os.makedirs(config.sample_dir)

    # Data loader.
    train_loader = get_loader(config.speakers_using, data_dir=config.train_data_dir, batch_size=config.batch_size, mode='train', num_workers=config.num_workers)
    test_loader = TestDataset(config.speakers_using, config.test_data_dir, config.wav_dir, src_spk='p225', trg_spk='p232')   # ********************************************

    # Solver for training and testing StarGAN.
    solver = Solver(train_loader, test_loader, config)

    if config.mode == 'train':    
        solver.train()

    # elif config.mode == 'test':
    #     solver.test()


if __name__ == '__main__':

    def config():
        pass


    config.speakers_using = cfg.speakers_used

    config.num_speakers = 70
    config.lambda_cls = 10
    config.lambda_rec = 50
    config.lambda_gp = 10
    config.sampling_rate = 16000
    config.batch_size = 100
    config.num_iters = 300000
    config.num_iters_decay = 100000
    config.g_lr = 0.0001
    config.d_lr = 0.0001
    config.c_lr = 0.0001
    config.n_critic = 5
    config.beta1 = 0.5
    config.beta2 = 0.999
    config.beta = 0.5
    config.gamma = 0.5
    config.resume_iters = 40000
    config.test_iters=100000
    config.num_workers=8
    config.mode='train'
    config.use_tensorboard=True
    config.train_data_dir=cfg.mc_train_path
    config.test_data_dir=cfg.mc_test_path
    config.wav_dir = cfg.wav16_path
    config.log_dir = cfg.logs_path
    config.model_save_dir = cfg.models_path
    config.sample_dir = './Voice_Conversion/samples'
    config.log_step = 10
    config.sample_step = 5000
    config.model_save_step = 15000
    config.lr_update_step=1000

    main(config)

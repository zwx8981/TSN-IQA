import argparse
import Train_all_bn
import scipy.io as sio
import os
from train_val_functions import *
import torch

def parse_config():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train", type=bool, default=True)
    parser.add_argument("--use_cuda", type=bool, default=True)
    parser.add_argument("--resume", type=bool, default=False)
    parser.add_argument("--resume_new", type=bool, default=False)
    parser.add_argument("--seed", type=int, default=19901116) 

    parser.add_argument("--backbone", type=str, default='resnet18') 
    parser.add_argument("--fc", type=bool, default=True)
    parser.add_argument('--scnn_root', type=str, default='saved_weights/scnn.pkl')

    parser.add_argument("--network", type=str, default="basecnn")  
    parser.add_argument("--representation", type=str, default="gap") 

    parser.add_argument("--ranking", type=bool, default=True)  # True for learning-to-rank False for regular regression
    parser.add_argument("--fidelity", type=bool,
                        default=True)  # True for fidelity loss False for regular ranknet with CE loss

    parser.add_argument("--resume_best", type=bool, default=True)
    parser.add_argument("--best_indicator", type=int, default=1) #1-6: LIVE, CSIQ, KADID10K, BID, CLIVE, KONIQ10K

    parser.add_argument("--split", type=int, default=1)
    parser.add_argument("--trainset", type=str, default="../IQA_Database")

    parser.add_argument("--live_set", type=str, default="../IQA_Database/databaserelease2/")
    parser.add_argument("--csiq_set", type=str, default="../IQA_Database/CSIQ/")
    parser.add_argument("--bid_set", type=str, default="../IQA_Database/BID/")
    parser.add_argument("--clive_set", type=str, default="../IQA_Database/ChallengeDB_release/")
    parser.add_argument("--koniq10k_set", type=str, default="../IQA_Database/koniq-10k/")
    parser.add_argument("--kadid10k_set", type=str, default="../IQA_Database/kadid10k/")

    parser.add_argument('--ckpt_path', default='./checkpoint', type=str,
                        metavar='PATH', help='path to checkpoints')
    parser.add_argument('--ckpt_resume_path', default='./checkpoint/', type=str,
                        metavar='PATH', help='path to checkpoints')
    parser.add_argument('--ckpt', default=None, type=str, help='name of the checkpoint to load')

    parser.add_argument("--train_txt", type=str,
                        default='train.txt')  

    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--batch_size2", type=int, default=32)
    parser.add_argument("--image_size", type=int, default=384, help='None means random resolution')
    parser.add_argument("--max_epochs", type=int, default=4)
    parser.add_argument("--max_epochs2", type=int, default=12)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--decay_interval", type=int, default=4)
    parser.add_argument("--decay_ratio", type=float, default=0.1)
    parser.add_argument("--epochs_per_eval", type=int, default=1)
    parser.add_argument("--epochs_per_save", type=int, default=1)

    parser.add_argument("--lwf", type=bool, default=False)
    parser.add_argument("--n_task", type=int, default=6)
    parser.add_argument("--task_id", type=int, default=0)
    parser.add_argument("--shared_head", type=bool, default=False)
    parser.add_argument("--get_scores", type=bool, default=False)

    parser.add_argument("--eval_each", type=bool, default=False)
    parser.add_argument("--enforce_bn", type=bool, default=False)

    parser.add_argument("--weighted_output", type=bool, default=False)

    parser.add_argument("--highest_output", type=bool, default=False) #set to False by default, 20231027

    parser.add_argument("--subfolder", type=str, default='train_on_live')

    parser.add_argument("--amp", type=bool, default=True)

    # kmeans related args
    parser.add_argument("--current_task_id", type=int, default=0)
    parser.add_argument('--base_ckpt_path', default='./checkpoint', type=str,
                        metavar='PATH', help='path to checkpoints')

    parser.add_argument("--kmeans", type=bool, default=False) # training with model
    parser.add_argument("--train_kmeans", type=int, default=0) # training after the model is trained

    parser.add_argument("--num_cluster", type=int, default=2)

    parser.add_argument("--reverse", type=bool, default=False)

    parser.add_argument("--verbose", type=bool, default=True)

    parser.add_argument("--reg_weight", type=float, default=10) #10 for lwf, 1000 for si, 10 for mas, 10000 for ewc

    parser.add_argument("--prev_weight", type=bool, default=False)

    parser.add_argument("--train_bn", type=bool, default=False)

    parser.add_argument("--b_fidelity", type=bool, default=True)

    parser.add_argument("--use_scnn", type=bool, default=False)

    parser.add_argument("--distortion_bn", type=bool, default=True)

    #other_methods
    parser.add_argument("--online_reg", type=bool, default=True)
    parser.add_argument("--reg_trigger", type=bool, default=False)
    #ewc
    parser.add_argument("--ewc", type=bool, default=False)
    parser.add_argument("--ewc_consolidate", type=bool, default=True)
    parser.add_argument("--ewc_online", type=bool, default=False)

    #SI
    parser.add_argument("--SI", type=bool, default=True)
    parser.add_argument("--si_consolidate", type=bool, default=False)
    parser.add_argument("--si_c", type=float, default=0.1)
    parser.add_argument("--epsilon", type=float, default=0.1)

    #MAS
    parser.add_argument("--MAS", type=bool, default=False)

    return parser.parse_args()


def main(cfg):
    t = Train_all_bn.Trainer(cfg)
    if cfg.train:
        t.fit()
    elif cfg.get_scores:
        with torch.no_grad():
            all_mos, all_hat = t.get_scores()
        scores_path = os.path.join('./scores/', cfg.subfolder)
        if not os.path.exists(scores_path):
            os.makedirs(scores_path)
        sio.savemat(os.path.join(scores_path, 'scores.mat'), {'mos': all_mos, 'hat': all_hat})
    elif cfg.train_kmeans:
        t.kmeans_each()
    elif cfg.train_expert:
        t.expert_each()
    else:
        if cfg.eval_each:
            if cfg.amp:
                from torch.cuda.amp import autocast as autocast
                with autocast():
                    test_results_srcc, val_results_srcc, all_hat_test, all_hat_val = t.eval_each()
            else:
                with torch.no_grad():
                    test_results_srcc, val_results_srcc, all_hat_test, all_hat_val = t.eval_each()

            out_str = 'Testing: LIVE SRCC: {:.4f}  CSIQ SRCC: {:.4f} KADID10K SRCC: {:.4f} \n' \
                      'BID SRCC: {:.4f} CLIVE SRCC: {:.4f}  KONIQ10K SRCC: {:.4f}'.format(
                test_results_srcc['live'],
                test_results_srcc['csiq'],
                test_results_srcc['kadid10k'],
                test_results_srcc['bid'],
                test_results_srcc['clive'],
                test_results_srcc['koniq10k'],
            )

            out_str2 = 'Validation: LIVE SRCC: {:.4f}  CSIQ SRCC: {:.4f} KADID10K SRCC: {:.4f} \n' \
                       'BID SRCC: {:.4f} CLIVE SRCC: {:.4f}  KONIQ10K SRCC: {:.4f}'.format(
                val_results_srcc['live'],
                val_results_srcc['csiq'],
                val_results_srcc['kadid10k'],
                val_results_srcc['bid'],
                val_results_srcc['clive'],
                val_results_srcc['koniq10k'],
            )
        else:
            with torch.no_grad():
                test_results_srcc, val_results_srcc, all_hat_test, all_hat_val = t.eval()
            out_str = 'Testing: LIVE SRCC: {:.4f}  CSIQ SRCC: {:.4f} KADID10K SRCC: {:.4f} \n' \
                      'BID SRCC: {:.4f} CLIVE SRCC: {:.4f}  KONIQ10K SRCC: {:.4f}'.format(
                test_results_srcc['live'],
                test_results_srcc['csiq'],
                test_results_srcc['kadid10k'],
                test_results_srcc['bid'],
                test_results_srcc['clive'],
                test_results_srcc['koniq10k'],
            )

            out_str2 = 'Validation: LIVE SRCC: {:.4f}  CSIQ SRCC: {:.4f} KADID10K SRCC: {:.4f} \n' \
                       'BID SRCC: {:.4f} CLIVE SRCC: {:.4f}  KONIQ10K SRCC: {:.4f}'.format(
                val_results_srcc['live'],
                val_results_srcc['csiq'],
                val_results_srcc['kadid10k'],
                val_results_srcc['bid'],
                val_results_srcc['clive'],
                val_results_srcc['koniq10k'],
            )


        print(out_str)
        print(out_str2)

        return test_results_srcc, val_results_srcc, all_hat_test, all_hat_val

method = 'TSN-IQA'
training = True
head_usage = 2 #0:last 1:oracle 2:kmeans 3:single

if __name__ == "__main__":

    if training:
        if method == 'SL':
            SL()
        elif method == 'LwF':
            LwF()
            train_kmeans_after_training()
            train_experts_after_training()
        elif method == 'TSN-IQA':
            TSN_IQA()
            train_kmeans_after_training()
            train_experts_after_training()
        else:
            Reg_CL()
            train_kmeans_after_training()
            train_experts_after_training()
    else:
        if head_usage == 0:
            eval_last_head()
        elif head_usage == 1:
            eval_each_head()
        elif head_usage == 2:
            eval_weight_head()
        else:
            eval_single_head()

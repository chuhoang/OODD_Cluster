# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import warnings
from numbers import Number

import mmcv
import numpy as np
import torch
import time
from mmcv import DictAction
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)
from mmcv.runner.checkpoint import save_checkpoint

from mmcls.apis import single_gpu_test_ood, single_gpu_test_ood_score, single_gpu_test_ssim
from mmcls.datasets import build_dataloader, build_dataset
from mmcls.models import build_ood_model
from mmcls.utils import get_root_logger, setup_multi_processes, gather_tensors, evaluate_all
from queue import PriorityQueue
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description='ood test')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    out_options = ['class_scores', 'pred_score', 'pred_label', 'pred_class']
    parser.add_argument(
        '--out-items',
        nargs='+',
        default=['all'],
        choices=out_options + ['none', 'all'],
        help='Besides metrics, what items will be included in the output '
        f'result file. You can choose some of ({", ".join(out_options)}), '
        'or use "all" to include all above, or use "none" to disable all of '
        'above. Defaults to output all.',
        metavar='')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results')
    parser.add_argument('--tmpdir', help='tmp dir for writing some results')
    parser.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='(Deprecated, please use --gpu-id) ids of gpus to use '
        '(only applicable to non-distributed testing)')
    parser.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='id of gpu to use '
        '(only applicable to non-distributed testing)')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument(
        '--device',
        choices=['cpu', 'cuda', 'ipu'],
        default='cuda',
        help='device used for testing')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args

def init_eval(cfg, args):
    # set multi-process settings
    setup_multi_processes(cfg)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None

    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    return cfg, distributed




class ScoreData:
    def __init__(self, score, data):
        self.score = score
        self.data = data
    
    def __lt__(self, other):
        return float(self.score) > float(other.score)

def kth_largest_per_column(matrix, k):
    """
    Compute the k-th largest element for each column of a matrix.
    """
    # top_values, _ = torch.topk(matrix, k, dim=0)
    # kth_values = top_values[-1, :]
    kth_values, _ = torch.kthvalue(matrix, matrix.size(0) - k + 1, dim=0)
    return kth_values.cpu().numpy()  

def batched_matrix_multiply(ftrain, second_matrix, K, batch_size=9000):
    """
    Compute the product of the feature tensors in batches.
    dimensions: ftrain: (n, d), second_matrix: (p, d), result: (n, p)
    then using the kth largest element in each column as the score
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ftrain_tensor = torch.tensor(ftrain, device=device)
    second_matrix_tensor = torch.tensor(second_matrix, device=device)
    p, _ = second_matrix_tensor.shape

    num_batches = p // batch_size + (p % batch_size > 0)
    res = []
    # for i in tqdm(range(num_batches), desc="Processing batches"):
    for i in range(num_batches):
        # Get the current batch of the second matrix
        start = i * batch_size
        end = min(start + batch_size, p)
        second_batch_tensor = second_matrix_tensor[start:end,:]
        batch_result = ftrain_tensor @ second_batch_tensor.T
        # batch_result = -torch.cdist(ftrain_tensor, second_batch_tensor, p=2)
        # Compute the k-th largest element for each column
        score = kth_largest_per_column(batch_result, K)
        # score = - sqrt(2(1-score))
        # score = -np.sqrt(2 * (1 - score))
        # score = score**2
        res.append(score)
    return np.concatenate(res, axis=0)


def main():
    args = parse_args()

    cfg = mmcv.Config.fromfile(args.config)

    if "multi_cfg" in cfg:
        multi_cfg = cfg.multi_cfg
    else:
        multi_cfg = [cfg]
    is_init = False

    for cfg in multi_cfg:
        if os.environ['LOCAL_RANK'] == '0':
            print("Evaluating {}...".format(cfg.readable_name))
        if not is_init:
            cfg, distributed = init_eval(cfg, args)
            is_init = True

        cfg.gpu_ids = [int(os.environ['LOCAL_RANK'])]
        print("cfg.data.id_data: ", cfg.data.id_data)
        dataset_id = build_dataset(cfg.data.id_data)
        dataset_ood = [build_dataset(d) for d in cfg.data.ood_data]
        name_ood = [d['name'] for d in cfg.data.ood_data]

        # build the dataloader
        # The default loader config
        loader_cfg = dict(
            # cfg.gpus will be ignored if distributed
            num_gpus=1 if args.device == 'ipu' else len(cfg.gpu_ids),
            dist=distributed,
            round_up=True,
        )
        # The overall dataloader settings
        loader_cfg.update({
            k: v
            for k, v in cfg.data.items() if k not in [
                'id_data', 'ood_data'
            ]
        })
        test_loader_cfg = {
            **loader_cfg,
            'shuffle': False,  # Not shuffle by default
            'sampler_cfg': None,  # Not use sampler by default
            **cfg.data.get('test_dataloader', {}),
        }
        # the extra round_up data will be removed during gpu/cpu collect
        data_loader_id = build_dataloader(dataset_id, **test_loader_cfg)
        data_loader_ood = []
        for ood_set in dataset_ood:
            data_loader_ood.append(build_dataloader(ood_set, **test_loader_cfg))

        model = build_ood_model(cfg.model)
        # if not cfg.model.classifier.type == 'VitClassifier':
        # model.init_weights()
        # model.classifier.backbone.change_weights()
        # if os.environ['LOCAL_RANK'] == '0':
        #     save_checkpoint(model.ood_detector.classifier, 'resnet50_random_block.pth')
        # assert False
        model = MMDataParallel(model, device_ids=cfg.gpu_ids)
        # model.to("cuda:{}".format(os.environ['LOCAL_RANK']))

        # init distributed env first, since logger depends on the dist info.
        # logger setup
        if os.environ['LOCAL_RANK'] == '0':
            timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
            log_file = os.path.join(cfg.work_dir, '{}_{}.log'.format(cfg.readable_name, timestamp))
            os.makedirs(cfg.work_dir, exist_ok=True)
            logger = get_root_logger(log_file=log_file, log_level=cfg.log_level,
                                     logger_name='mmcls' if len(multi_cfg) == 1 else cfg.readable_name)
        if os.environ['LOCAL_RANK'] == '0':
            print()
            print("Processing in-distribution data...")

        ## new add
        cache_dir = "./cache"
        cache_id_score = os.path.join(cache_dir, 'id.npy')
        cache_id_feat = os.path.join(cache_dir, 'id_feat.npy')

        # if not os.path.exists(cache_id_score):
        if True:
            outputs_id, id_feats = single_gpu_test_ood(model, data_loader_id, 'ID')
            id_score = outputs_id
            id_feat = id_feats
            np.save(cache_id_score, id_score)
            np.save(cache_id_feat, id_feat)
            print(f"save to cache {cache_id_score}")
            print(f"save to cache {cache_id_feat}")
        else:
            id_score = np.load(cache_id_score)
            id_feat = np.load(cache_id_feat)
            print(f"load from cache {cache_id_score}")
            print(f"load from cache {cache_id_feat}")
        # print("outputs_id: ", outputs_id.shape)
        # in_scores = gather_tensors(outputs_id)
        # in_scores = outputs_id
        # in_scores = np.concatenate(in_scores, axis=0)
        # if os.environ['LOCAL_RANK'] == '0':
        #     print("Average ID score:", in_scores.mean())
          
        result_list = []
        for ood_set, ood_name in zip(data_loader_ood, name_ood):
            if os.environ['LOCAL_RANK'] == '0':
                print()
                print("Processing out-of-distribution data ({})...".format(ood_name))
            cache_out_score = os.path.join(cache_dir, ood_name+'_id.npy')
            cache_out_feat = os.path.join(cache_dir, ood_name+'_id_feat.npy')
            if True:
                outputs_ood, ood_feats = single_gpu_test_ood(model, ood_set, ood_name)
                out_score, out_feat = outputs_ood, ood_feats
                np.save(cache_out_score, out_score)
                print(f"save to cache {cache_out_score}")
                np.save(cache_out_feat, out_feat)
                print(f"save to cache {cache_out_feat}")
            else:
                print(f"load from cache {cache_out_score}")
                out_score = np.load(cache_out_score)
                out_feat = np.load(cache_out_feat)
                print(f"load from cache {cache_out_feat}")    
            # out_scores = gather_tensors(outputs_ood)
            # out_scores = np.concatenate(out_scores, axis=0)
            # out_scores = outputs_ood

            all_output = np.concatenate([id_score, out_score], axis=0)
            label_id = np.concatenate([np.ones(id_feat.shape[0]), np.zeros(out_feat.shape[0])], axis=0)  
            np.random.seed(110)     
            idx = np.random.permutation(all_output.shape[0])
           
            label_id = label_id[idx] 
            all_output = all_output[idx]

            batch_size = 128
            
            num_batches = all_output.shape[0] // batch_size + (all_output.shape[0] % batch_size > 0)
        
            scores_list = []

            for i in tqdm(range(num_batches), desc="Continue Learning"):
                start = i * batch_size
                end = min(start + batch_size, all_output.shape[0])
                batch_score =  all_output[start:end]
                    
                # print(f"batch_score: {batch_score}")
                scores_list.append(batch_score)
            scores_all = np.concatenate(scores_list, axis=0)

        
            scores_in_final = scores_all[label_id == 1]
            print(f"len of in scores: {scores_in_final.shape}")
            scores_ood_final = scores_all[label_id == 0]
            print(f"len of out scores: {scores_ood_final.shape}")

          
            in_scores = scores_in_final
            out_scores = scores_ood_final

            if os.environ['LOCAL_RANK'] == '0':
                print("Average OOD {} score:".format(ood_name), out_scores.mean())
            if os.environ['LOCAL_RANK'] == '0':
                auroc, aupr_in, aupr_out, fpr95 = evaluate_all(in_scores, out_scores)
                result_list.extend([auroc, aupr_in, aupr_out, fpr95])
                logger.critical('============Overall Results for {}============'.format(ood_name))
                logger.critical('AUROC: {}'.format(auroc))
                logger.critical('AUPR (In): {}'.format(aupr_in))
                logger.critical('AUPR (Out): {}'.format(aupr_out))
                logger.critical('FPR95: {}'.format(fpr95))
                logger.critical('quick data: {},{},{},{}'.format(auroc, aupr_in, aupr_out, fpr95))
               
        avg_auroc = 0
        avg_fpr95 = 0
        if os.environ['LOCAL_RANK'] == '0':
            for idx in range(len(data_loader_ood)):
                avg_auroc += result_list[4 * idx + 0] / len(data_loader_ood)
                avg_fpr95 += result_list[4 * idx + 3] / len(data_loader_ood)
            result_list.extend([avg_auroc, avg_fpr95])
            logger.critical('all quick data: '+",".join(list(map(str, result_list))))

if __name__ == '__main__':
    main()

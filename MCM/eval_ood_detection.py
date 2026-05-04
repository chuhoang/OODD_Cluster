import os
import argparse
import numpy as np
import torch
from scipy import stats

from utils.common import setup_seed, get_num_cls, get_test_labels
from utils.detection_util import get_Mahalanobis_score, get_mean_prec, print_measures, get_and_print_results, get_ood_scores_clip
from utils.file_ops import save_as_dataframe, setup_log
from utils.plot_util import plot_distribution
from utils.train_eval_util import  set_model_clip, set_train_loader, set_val_loader, set_ood_loader_ImageNet
# sys.path.append(os.path.dirname(__file__))
from queue import PriorityQueue
from tqdm import tqdm

def process_args():
    parser = argparse.ArgumentParser(description='Evaluates MCM Score for CLIP',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # setting for each run
    parser.add_argument('--in_dataset', default='ImageNet', type=str,
                        choices=['ImageNet', 'ImageNet10', 'ImageNet20', 'ImageNet100',
                                  'pet37', 'food101', 'car196', 'bird200'], help='in-distribution dataset')
    parser.add_argument('--root-dir', default="datasets", type=str,
                        help='root dir of datasets')
    parser.add_argument('--name', default="eval_ood",
                        type=str, help="unique ID for the run")
    parser.add_argument('--seed', default=5, type=int, help="random seed")
    parser.add_argument('--gpu', default=0, type = int,
                        help='the GPU indice to use')
    parser.add_argument('-b', '--batch-size', default=512, type=int,
                        help='mini-batch size')
    parser.add_argument('--T', type=int, default=1,
                        help='temperature parameter')
    parser.add_argument('--model', default='CLIP', type=str, help='model architecture')
    parser.add_argument('--CLIP_ckpt', type=str, default='ViT-B/16',
                        choices=['ViT-B/32', 'ViT-B/16', 'ViT-L/14'], help='which pretrained img encoder to use')
    parser.add_argument('--ckpt', type=str, default='./clip-vit-base-patch16',
                        choices=['ViT-B/32', 'ViT-B/16', 'ViT-L/14'], help='which pretrained img encoder to use')
    parser.add_argument('--score', default='MCM', type=str, choices=[
        'MCM', 'energy', 'max-logit', 'entropy', 'var', 'maha'], help='score options')
    parser.add_argument('--mp', type=float, default=0.95,)
    parser.add_argument('--k', type=int, default=200,)
    args = parser.parse_args()

    # for Mahalanobis score
    parser.add_argument('--feat_dim', type=int, default=512, help='feat dim； 512 for ViT-B and 768 for ViT-L')
    parser.add_argument('--normalize', type = bool, default = False, help='whether use normalized features for Maha score')
    parser.add_argument('--generate', type = bool, default = True, help='whether to generate class-wise means or read from files for Maha score')
    parser.add_argument('--template_dir', type = str, default = 'img_templates', help='the loc of stored classwise mean and precision matrix')
    parser.add_argument('--subset', default = False, type =bool, help = "whether uses a subset of samples in the training set")
    parser.add_argument('--max_count', default = 250, type =int, help = "how many samples are used to estimate classwise mean and precision matrix")
    args = parser.parse_args()

    args.n_cls = get_num_cls(args)
    args.log_directory = f"results/{args.in_dataset}/{args.score}/{args.model}_{args.CLIP_ckpt}_T_{args.T}_ID_{args.name}"
    os.makedirs(args.log_directory, exist_ok=True)

    return args

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
    args = process_args()
    setup_seed(args.seed)
    # log = setup_log(args)
    log = None
    assert torch.cuda.is_available()
    torch.cuda.set_device(args.gpu)

    net, preprocess = set_model_clip(args)
    net.eval()

    if args.in_dataset in ['ImageNet10']: 
        out_datasets = ['ImageNet20']
    elif args.in_dataset in ['ImageNet20']: 
        out_datasets = ['ImageNet10']
    elif args.in_dataset in [ 'ImageNet', 'ImageNet100', 'bird200', 'car196', 'food101', 'pet37']:
         out_datasets = ['iNaturalist','SUN', 'places365', 'dtd']
    test_loader = set_val_loader(args, preprocess)
    test_labels = get_test_labels(args, test_loader)

    ## new add
    cache_dir = "./cache"
    cache_id_score = os.path.join(cache_dir, 'id.npy')
    cache_id_feat = os.path.join(cache_dir, 'id_feat.npy')


  
    if not os.path.exists(cache_id_score):
        id_score, id_feat = get_ood_scores_clip(args, net, test_loader, test_labels, in_dist=True)
        np.save(cache_id_score, id_score)
        np.save(cache_id_feat, id_feat)
        print(f"save to cache {cache_id_score}")
        print(f"save to cache {cache_id_feat}")
    else:
        print(f"load from cache {cache_id_score}")
        print(f"load from cache {cache_id_feat}")
        id_score = np.load(cache_id_score)
        id_feat = np.load(cache_id_feat)

    auroc_list, aupr_list, fpr_list = [], [], []
    print("dataset: ", out_datasets)

    for out_dataset in out_datasets[:]:

        # log.debug(f"Evaluting OOD dataset {out_dataset}")
        ood_loader = set_ood_loader_ImageNet(args, out_dataset, preprocess, root=os.path.join(args.root_dir, 'ImageNet_OOD_dataset'))

        cache_out_score = os.path.join(cache_dir, out_dataset+'_id.npy')
        cache_out_feat = os.path.join(cache_dir, out_dataset+'_id_feat.npy')
        if not os.path.exists(cache_out_score):
            out_score, out_feat = get_ood_scores_clip(args, net, ood_loader, test_labels)
            np.save(cache_out_score, out_score)
            print(f"save to cache {cache_out_score}")
            np.save(cache_out_feat, out_feat)
            print(f"save to cache {cache_out_feat}")
        else:
            print(f"load from cache {cache_out_score}")
            out_score = np.load(cache_out_score)
            out_feat = np.load(cache_out_feat)
            print(f"load from cache {cache_out_feat}")

        ## new add--OODD
        queue_size =2048 # 2048
        queue = PriorityQueue()
        all_data = np.concatenate([id_feat, out_feat], axis=0)

        all_output = np.concatenate([id_score, out_score], axis=0)
        label_id = np.concatenate([np.ones(id_feat.shape[0]), np.zeros(out_feat.shape[0])], axis=0)  
        np.random.seed(110)     
        idx = np.random.permutation(all_data.shape[0])
        all_data = all_data[idx]
        label_id = label_id[idx] 
        all_output = all_output[idx]

        batch_size = 64 
        
        num_batches = all_data.shape[0] // batch_size + (all_data.shape[0] % batch_size > 0)
    
        scores_list = []

        for i in tqdm(range(num_batches), desc="Continue Learning"):
            start = i * batch_size
            end = min(start + batch_size, all_data.shape[0])
            batch_data = all_data[start:end]
            batch_score = - all_output[start:end]
                   
            for j in range(batch_score.shape[0]):
                queue.put(ScoreData(batch_score[j], batch_data[j]))
                if queue.qsize() > queue_size:
                    queue.get()

            data_list = []
            for item in list(queue.queue):
                data_list.append(item.data)
            new_food = np.array(data_list)
            ood_batch_score = batched_matrix_multiply(new_food, batch_data, 5) 
            batch_score = batch_score - ood_batch_score 
            scores_list.append(batch_score)
        scores_all = np.concatenate(scores_list, axis=0)

     
        scores_in_final = scores_all[label_id == 1]
        print(f"len of in scores: {scores_in_final.shape}")
        scores_ood_final = scores_all[label_id == 0]
        print(f"len of out scores: {scores_ood_final.shape}")

        while queue.qsize() > 0:
            queue.get()

        del queue, all_output, label_id, idx, all_data, out_feat, scores_list, scores_all
        id_score_ = - scores_in_final
        out_score_ = - scores_ood_final
        ## OODD
       
        plot_distribution(args, id_score_, out_score_, out_dataset)
        
        get_and_print_results(args, log, id_score_, out_score_,
                              auroc_list, aupr_list, fpr_list)
    
    print(("mean score"))
    print_measures(log, np.mean(auroc_list), np.mean(aupr_list),
                   np.mean(fpr_list), method_name=args.score)
    save_as_dataframe(args, out_datasets, fpr_list, auroc_list, aupr_list)


if __name__ == '__main__':
    main()

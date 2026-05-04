from openood.postprocessors import KNNPostprocessor
from sklearn import metrics
import numpy as np
import torch
def test():
    idx = np.load('train_idx.npy')
    knn = KNNPostprocessor(K1 = 5,
        K2 = 10,
        ALPHA = 0.5,
        queue_size = 200,
        activation_log = None,
        id_feature = None,
        setup_flag = False,
        id_name = None,
        aux_feature = None,
        count = 0,
        idx = idx
    )
    # a = np.load('/home/hoangcm/Signlanguage/feat_ood_val_contrast.npy')
    # b = np.load('/home/hoangcm/Signlanguage/feat_id_val_contrast.npy')
    
    food = np.load('/home/hoangcm/Signlanguage/feat_ood_val_contrast.npy').squeeze(-1).squeeze(-1)
    ftest = np.load('/home/hoangcm/Signlanguage/feat_id_val_contrast.npy').squeeze(-1).squeeze(-1)
    scores_in_final, scores_ood_final, label_id = knn.conf_postprocess(food, ftest) 
    conf = np.concatenate([scores_in_final, scores_ood_final], axis=0)
    ood_indicator = np.zeros_like(label_id)
    ood_indicator[label_id == 0] = 1
    fpr_list, tpr_list, thresholds = metrics.roc_curve(ood_indicator, -conf)
    auroc = metrics.auc(fpr_list, tpr_list)
    print(f"AUROC: {auroc:.4f}")

def filter_activation_log():
    msp_list = np.load('/home/hoangcm/Signlanguage/conf_id_train_constrast.npy')
    label_list = np.load('/home/hoangcm/Signlanguage/label_id_train_contrast.npy')
    msp_list = torch.from_numpy(msp_list)
    label_list = torch.from_numpy(label_list)

    unique_classes = torch.unique(label_list)
  
    class_sorted_indices = []
    max_class_samples = 0
    
    for cls in unique_classes:
        # Get indices of samples belonging to this class
        cls_indices = torch.where(label_list == cls)[0]
        
        # Get the confidence scores for these samples
        cls_msp = msp_list[cls_indices]
        
        # Sort these indices by confidence in descending order
        sorted_cls_indices = torch.argsort(cls_msp, descending=True)
        cls_indices = cls_indices.to(sorted_cls_indices.device)
        # Convert back to original indices
        sorted_orig_indices = cls_indices[sorted_cls_indices].cpu().numpy()
        class_sorted_indices.append(sorted_orig_indices)
        max_class_samples = max(max_class_samples, len(sorted_orig_indices))
    
    final_idx = []
    
    # Interleave the classes: first highest confidence from each class, then second highest, etc.
    for i in range(max_class_samples):
        for class_idx, sorted_indices in enumerate(class_sorted_indices):
            if i < len(sorted_indices):
                final_idx.append(sorted_indices[i])
    
    idx = np.array(final_idx, dtype=int)
    np.save('train_idx.npy', idx)
filter_activation_log()
test()
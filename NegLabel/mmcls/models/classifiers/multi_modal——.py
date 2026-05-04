import os
import torch
from torch import nn
import xml.etree.ElementTree as ET
import random
import numpy as np

import clip
import time
from torchvision.datasets import CIFAR100, CIFAR10
from ..builder import CLASSIFIERS, build_backbone, build_head, build_neck
from ..heads import MultiLabelClsHead
from ..utils.augment import Augments
from .base import BaseClassifier
from .class_names import CLASS_NAME, prompt_templates, adj_prompts_templetes

def chunked_quantile_mask_per_chunk(data: torch.Tensor, mp_: float, chunk_size: int):
    mask_list = []
    N = data.shape[0]

    for start_idx in range(0, N, chunk_size):
        end_idx = min(start_idx + chunk_size, N)
        # 当前 chunk
        data_chunk = data[start_idx:end_idx]  # shape: [chunk_size, D]

        # 对 chunk_size 条数据(在 dim=1 上)计算分位数
        # 如果你是想对每一行单独计算分位数，可再结合 for 循环或其他逻辑
        q_chunk = torch.quantile(data_chunk.float(), mp_, dim=1, keepdim=True)  
        
        # 得到 mask
        mask_chunk = data_chunk > q_chunk
        mask_list.append(mask_chunk)

    # 拼接
    mask = torch.cat(mask_list, dim=0)
    return mask

@CLASSIFIERS.register_module()
class CLIPScalableClassifier(BaseClassifier):

    def __init__(self,
                 arch='ViT-B/16',
                 train_dataset=None,
                 wordnet_database=None,
                 txt_exclude=None,
                 neg_subsample=-1,
                 neg_topk=10000,
                 emb_batchsize=1000,
                 init_cfg=None,
                 prompt_idx_pos=None,
                 prompt_idx_neg=None,
                 exclude_super_class=None,
                 dump_neg=False,
                 cls_mode=False,
                 load_dump_neg=False,
                 pencentile=1,
                 pos_topk=None,):
        super(CLIPScalableClassifier, self).__init__(init_cfg)
        self.local_rank = os.environ['LOCAL_RANK']
        self.device = "cuda:{}".format(self.local_rank)

        self.clip_model, _ = clip.load(arch, self.device, jit=False)
        self.clip_model.eval()
        self.model = self.clip_model
        self.cls_mode = cls_mode


        if prompt_idx_pos is None:
            prompt_idx_pos = -1
        if exclude_super_class is not None:
            class_name=CLASS_NAME[train_dataset][exclude_super_class]
        else:
            class_name=CLASS_NAME[train_dataset]
        prompts = [prompt_templates[prompt_idx_pos].format(c) for c in class_name]
        text_inputs_pos = torch.cat([clip.tokenize(f"{c}") for c in prompts]).to(self.device)

        with torch.no_grad():
            self.text_features_pos = self.clip_model.encode_text(text_inputs_pos).to(torch.float32)
            self.text_features_pos /= self.text_features_pos.norm(dim=-1, keepdim=True)

        if not load_dump_neg or not os.path.exists('/data/neg_label/neg_embedding/neg_dump.pth'):
            txtfiles = os.listdir(wordnet_database)
            if txt_exclude:
                file_names = txt_exclude.split(',')
                for file in file_names:
                    txtfiles.remove(file)
            words_noun = []
            words_adj = []
            if prompt_idx_neg is None:
                prompt_idx_neg = -1
            prompt_templete = dict(
                    adj='This is a {} photo',
                    noun=prompt_templates[prompt_idx_neg],
                )
            dedup = dict()
            for file in txtfiles:
                filetype = file.split('.')[0]
                if filetype not in prompt_templete:
                    continue
                with open(os.path.join(wordnet_database, file), 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        if line.strip() in dedup:
                            continue
                        dedup[line.strip()] = None
                        if filetype == 'noun':
                            if pos_topk is not None:
                                if line.strip() in class_name:
                                    continue
                            words_noun.append(prompt_templete[filetype].format(line.strip()))
                        elif filetype == 'adj':
                            words_adj.append(prompt_templete[filetype].format(line.strip()))
                        else:
                            raise TypeError

            if neg_subsample > 0:
                random.seed(42)
                words_noun = random.sample(words_noun, neg_subsample)

            text_inputs_neg_noun = torch.cat([clip.tokenize(f"{c}") for c in words_noun]).to(self.device)
            text_inputs_neg_adj = torch.cat([clip.tokenize(f"{c}") for c in words_adj]).to(self.device)
            text_inputs_neg = torch.cat([text_inputs_neg_noun, text_inputs_neg_adj], dim=0)
            noun_length=len(text_inputs_neg_noun)
            adj_length = len(text_inputs_neg_adj)


            with torch.no_grad():
                self.text_features_neg = []
                for i in range(0, len(text_inputs_neg), emb_batchsize):
                    x = self.clip_model.encode_text(text_inputs_neg[i : i + emb_batchsize])
                    self.text_features_neg.append(x)
                self.text_features_neg = torch.cat(self.text_features_neg, dim=0)
                self.text_features_neg /= self.text_features_neg.norm(dim=-1, keepdim=True)
                if dump_neg:
                    tmp = self.text_features_neg.cpu()
                    dump_dict=dict(neg_emb=tmp, noun_length=noun_length, adj_length=adj_length)
                    os.makedirs('/data/neg_label/neg_embedding', exist_ok=True)
                    torch.save(dump_dict, '/data/neg_label/neg_embedding/neg_dump.pth')
                    assert False
        else:
            tic = time.time()
            dump_dict = torch.load('/data/neg_label/neg_embedding/neg_dump.pth')
            self.text_features_neg = dump_dict['neg_emb'].to(self.device)
            toc = time.time()
            print('Successfully load the negative embedding and cost {}s.'.format(toc-tic))
            noun_length = dump_dict['noun_length']
            adj_length = dump_dict['adj_length']

        with torch.no_grad():
            self.text_features_neg = self.text_features_neg.to(torch.float32)

            if pos_topk is not None:
                pos_mask = torch.zeros(len(self.text_features_neg), dtype=torch.bool, device=self.device)
                for i in range(self.text_features_pos.shape[0]):
                    sim = self.text_features_pos[i].unsqueeze(0) @ self.text_features_neg.T
                    _, ind = torch.topk(sim.squeeze(0), k=pos_topk)
                    pos_mask[ind] = 1
                self.text_features_pos = torch.cat([self.text_features_pos, self.text_features_neg[pos_mask]])

            neg_sim = []
            for i in range(0, noun_length+adj_length, emb_batchsize):
                tmp = self.text_features_neg[i: i + emb_batchsize] @ self.text_features_pos.T
                tmp = tmp.to(torch.float32)
                sim = torch.quantile(tmp, q=pencentile, dim=-1)
                neg_sim.append(sim)
            neg_sim = torch.cat(neg_sim, dim=0)
            neg_sim_noun = neg_sim[:noun_length]
            neg_sim_adj = neg_sim[noun_length:]
            text_features_neg_noun = self.text_features_neg[:noun_length]
            text_features_neg_adj = self.text_features_neg[noun_length:]

            ind_noun = torch.argsort(neg_sim_noun)
            ind_adj = torch.argsort(neg_sim_adj)


            self.text_features_neg = torch.cat([text_features_neg_noun[ind_noun[0:int(len(ind_noun)*neg_topk)]],
                                                text_features_neg_adj[ind_adj[0:int(len(ind_adj)*neg_topk)]]], dim=0)


            self.adj_start_idx = int(len(ind_noun) * neg_topk)

            ## If you want to dump the selected negative labels (with prompt), please uncomment these lines.
            # with open("selected_neg_labels.txt", "w") as f:
            #     for i in ind_noun[0:int(len(ind_noun)*neg_topk)]:
            #         f.write("{}\n".format(words_noun[i]))
            #     for j in ind_adj[0:int(len(ind_adj)*neg_topk)]:
            #         f.write("{}\n".format(words_adj[j]))
    
    def extract_feat(self, img, stage='neck'):
        raise NotImplementedError

    def forward_train(self, img, gt_label, **kwargs):
        raise NotImplementedError

    def simple_test(self, img, img_metas=None, require_features=False, require_backbone_features=False, softmax=True, **kwargs):
        """Test without augmentation."""
        with torch.no_grad():
            image_features = self.model.encode_image(img)
            image_features /= image_features.norm(dim=-1, keepdim=True) + 1e-10

        if self.cls_mode:
            image_features = image_features.to(torch.float32)
            self.text_features_pos = self.text_features_pos.to(torch.float32)
            pos_sim = (100.0 * image_features @ self.text_features_pos.T)
            pos_sim = list(pos_sim.softmax(dim=-1).detach().cpu().numpy())            
            return pos_sim
        else:
            image_features = image_features.to(torch.float32)
            self.text_features_pos = self.text_features_pos.to(torch.float32)
            self.text_features_neg = self.text_features_neg.to(torch.float32)
            pos_sim = (100.0 * image_features @ self.text_features_pos.T)
            neg_sim = (100.0 * image_features @ self.text_features_neg[:10000].T)




            
            logit_scale = 100.0
        
            
            # text_features = torch.cat([self.text_features_pos, self.text_features_neg], dim=0)
            k = 300
            logits = logit_scale * image_features @ self.text_features_pos.t()
            pred_label = torch.topk(logits, k)[1]
            data = (image_features.unsqueeze(1) * self.text_features_pos[pred_label])

            mp_ = 0.8
            mask = data > torch.quantile(data.float(), mp_, 1, True)
            # mask = chunked_quantile_mask_per_chunk(data, mp_, 100)

            content_embd = torch.where(mask, image_features.unsqueeze(1).repeat(1, k, 1), 0)
            
            env_embd = image_features.unsqueeze(1).repeat(1, k, 1) - content_embd

            pos_logits_env = logit_scale * env_embd @ self.text_features_pos.t()


            neg_logits_env_list = []
            N = self.text_features_neg.shape[0]
            chunk_size = 1000
            for start_idx in range(0, N, chunk_size):
                end_idx = min(start_idx + chunk_size, N)
                # 当前 chunk
                if end_idx - start_idx < chunk_size:
                    break
                text_features_neg_chunk = self.text_features_neg[start_idx:end_idx]
                logits = logit_scale * image_features @ text_features_neg_chunk.t()
                pred_label = torch.topk(logits, k)[1]
                data = (image_features.unsqueeze(1) * text_features_neg_chunk[pred_label])

                mp_ = 0.8
                mask = data > torch.quantile(data.float(), mp_, 1, True)
                # mask = chunked_quantile_mask_per_chunk(data, mp_, 100)

                content_embd = torch.where(mask, image_features.unsqueeze(1).repeat(1, k, 1), 0)
                
                env_embd = image_features.unsqueeze(1).repeat(1, k, 1) - content_embd

                neg_logits_env_i = logit_scale * env_embd @ self.text_features_pos.t()
                neg_logits_env_list.append(neg_logits_env_i)

            # 拼接
            neg_logits_env = torch.cat(neg_logits_env_list, dim=-1)
            del neg_logits_env_list
            print("neg_logits_env", neg_logits_env.shape)
         
            ################################################################################
            ################################################################################
  
            # energy = -torch.logsumexp(logits, -1).unsqueeze(1).repeat(1, k)
            # energy_env = -torch.logsumexp(logits_env, -1)
            # print("energy", energy.shape)
            # print("energy_env", energy_env.shape)
            # delta_env = (energy_env - energy).mean(1)
        

            return pos_sim, neg_sim, pos_logits_env, neg_logits_env,image_features






            return pos_sim, neg_sim, image_features

from transformers import AutoTokenizer
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertEncoder
from transformers.file_utils import ModelOutput
from torch.nn import CrossEntropyLoss, MSELoss
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict, Counter
from .graph import GraphEncoder
from .hierarchy_consistency import HierarchicalTripletLoss
from .bert import BertModel, BertPreTrainedModel, BertEmbeddings, BertPoolingLayer, BertOutputLayer
from .text_attention import generate
from utils import get_hierarchy_info
import pickle
import random
import os


def multilabel_categorical_crossentropy(y_true, y_pred):
    loss_mask = y_true != -100
    y_true = y_true.masked_select(loss_mask).view(-1, y_pred.size(-1))
    y_pred = y_pred.masked_select(loss_mask).view(-1, y_true.size(-1))
    y_pred = (1 - 2 * y_true) * y_pred
    y_pred_neg = y_pred - y_true * 1e12
    y_pred_pos = y_pred - (1 - y_true) * 1e12
    zeros = torch.zeros_like(y_pred[:, :1])
    y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
    y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
    neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
    pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
    return (neg_loss + pos_loss).mean()


class MultiAttBlock(nn.Module):
    def __init__(self, embed_dim,
                 num_heads,
                 qdim=None,
                 kdim=None,
                 dropout=0.3,
    ):
        # qdim: num_of_labels
        # kdim: seq_len
        super(MultiAttBlock, self).__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.embed_dim = embed_dim

        self.q_embed_size = qdim if qdim else embed_dim
        self.k_embed_size = kdim if kdim else embed_dim
        self.v_embed_size = kdim if kdim else embed_dim
        self.dropout = dropout
        self.head_scale = self.head_dim ** -0.5

        # modify the size here
        d_hq = embed_dim // num_heads
        d_hv = embed_dim // num_heads

        # Define the query, key, and value linear transformations for each head
        self.query_heads = nn.Linear(self.q_embed_size, self.q_embed_size, bias=True)
        self.key_heads = nn.Linear(self.k_embed_size, self.k_embed_size, bias=True)
        self.value_heads = nn.Linear(self.v_embed_size, self.v_embed_size, bias=True)

        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)   # 最后的输出投影层
        self.query_block = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.reset_parameters()

    # 重置模型中的所有线性层的权重和偏置
    def reset_parameters(self):
        self.query_heads.reset_parameters()
        self.key_heads.reset_parameters()
        self.value_heads.reset_parameters()

        self.out_proj.reset_parameters()
        self.query_block.reset_parameters()

    def multiAttn(self, Q, K, V, key_padding_mask=None):
        # Q: batch_size * num_labels * embed_dim
        # K: batch_size * seq_len * embed_dim
        # V: batch_size * seq_len * embed_dim
        # key_padding_mask: batch_size * seq_len  形状为[batch_size,seq_len]，用于指示哪些位置是有效的掩码

        # 将输入的查询、键和值分别通过线性层映射到注意力空间
        Q_proj = self.query_heads(Q) # batch_size * num_labels * embed_dim
        K_proj = self.key_heads(K) # batch_size * seq_len * embed_dim
        V_proj = self.value_heads(V) # batch_size * seq_len * embed_dim

        bsz, num_labels, embed_dim = Q_proj.shape
        bsz, seq_len, embed_dim = K_proj.shape

        # 将查询、键和值分别按照多头的方式分割
        Q_proj = Q_proj.transpose(0, 1).reshape(num_labels, bsz * self.num_heads, self.head_dim).transpose(0, 1) # (bsz * num_heads) * num_labels * d_hq
        K_proj = K_proj.transpose(0, 1).reshape(seq_len, bsz * self.num_heads, self.head_dim).transpose(0, 1) # (bsz * num_heads) * seq_len * d_hq
        V_proj = V_proj.transpose(0, 1).reshape(seq_len, bsz * self.num_heads, self.head_dim).transpose(0, 1) # (bsz * num_heads) * seq_len * d_hv

        # 点积计算查询和键之间的相似性，得到注意力得分，使用缩放因子 self.head_scale 防止数值过大，影响 Softmax 稳定性。
        scores = torch.bmm(Q_proj, K_proj.transpose(-2, -1)) * self.head_scale # (bsz * num_heads) * num_labels * seq_len
        
        # check if the scores between different batches are the same
        # print(np.isclose(scores[0].cpu().detach().numpy(), scores[self.num_heads].cpu().detach().numpy()).all())

        # 如果提供了 key_padding_mask，将非有效位置的得分设置为 -inf，以便在后续 Softmax 中忽略这些位置
        if key_padding_mask is not None:
            # Reshape the key_padding_mask to have shape (bsz * num_heads, 1, seq_len) to enable broadcasting
            scores = scores.view(bsz, self.num_heads, num_labels, seq_len)
            # filp the mask
            key_padding_mask = key_padding_mask.eq(0)
            key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(2) # bsz * 1 * 1 * seq_len
            scores = scores.masked_fill(key_padding_mask, float('-inf'))
            scores = scores.view(bsz * self.num_heads, num_labels, seq_len)

        attn_weights = F.softmax(scores, dim=-1) # (bsz * num_heads) * num_labels * seq_len
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)

        # 根据注意力权重和值向量计算加权输出
        attn = torch.bmm(attn_weights, V_proj) # (bsz * num_heads) * num_labels * d_hv
        # 将多头的输出重新组合成原始维度
        attn = attn.transpose(0, 1).reshape(num_labels, bsz, self.embed_dim).transpose(0, 1) # bsz * num_labels * embed_dim

        # check if the attention between different batches are the same
        # print(np.isclose(attn[0].cpu().detach().numpy(), attn[1].cpu().detach().numpy()).all())
        attn = self.out_proj(attn) # bsz * num_labels * embed_dim  进行输出投影

        attn_weights = attn_weights.view(bsz, self.num_heads, num_labels, seq_len)

        return attn, attn_weights
    
    def forward(self, Q, K, V, key_padding_mask=None, need_weights=False, attn_mask=None):
        # Q: batch_size * num_labels * embed_dim
        # K: batch_size * seq_len * embed_dim
        # V: batch_size * seq_len * embed_dim

        # 调用 multiAttn 计算注意力输出 _Q 和注意力权重 attns
        _Q, attns = self.multiAttn(Q, K, V, key_padding_mask=key_padding_mask)
        # 将 _Q 加上查询的原始投影 self.query_heads(Q)，实现残差连接
        _Q += self.query_heads(Q)

        # 再通过额外的查询增强层 self.query_block，进一步优化注意力输出
        return _Q + self.query_block(_Q), attns


class LabelAware(nn.Module): # Compute the label aware embedding
    def __init__(self, label_embedding_size, attn_hidden_size, head):
        super(LabelAware, self).__init__()
        self.label_embedding_size = label_embedding_size  # 标签表示的嵌入维度大小（label_repr 的最后一维）
        self.attn_hidden_size = attn_hidden_size  # 注意力机制中的隐藏维度
        # 多头注意力块，用于计算 label_repr 和 input_data 的交互关系
        self.multi_attn_block = MultiAttBlock(self.label_embedding_size, head, self.label_embedding_size, self.label_embedding_size)

    def forward(self, input_data, label_repr, input_data_mask=None, label_repr_mask=None):
        # input_data: batch_size * seq_len * hidden_size
        # label_repr: batch_size * num_labels * label_embedding_size
        # input_data_mask: batch_size * seq_len  输入特征的掩码，用于指示序列中哪些位置是有效的
        # label_repr_mask：batch_size * num_labels  标签的掩码，指示哪些标签是有效的

        # 对输入特征 input_data 和标签表示 label_repr 的无效部分进行屏蔽
        if input_data_mask is not None:
            input_data = input_data * input_data_mask.unsqueeze(-1)
        if label_repr_mask is not None:
            label_repr = label_repr * label_repr_mask.unsqueeze(-1)

        # 通过 label_repr（作为查询向量）和 input_data（作为键和值向量）之间的注意力机制，生成标签感知的嵌入表示
        label_aware, attns = self.multi_attn_block(label_repr, input_data, input_data, input_data_mask)

        # label_aware：batch_size * num_labels * label_embedding_size 标签感知嵌入
        # attns：batch_size * num_heads * num_labels * seq_len  注意力权重，表示标签与输入特征之间的注意力分布
        return label_aware, attns

    def update_history(self):
        if hasattr(self, '_pending_update'):
            current = self._pending_update['current']
            self.prev_step_embedding = current

            # 按标签独立累积全局历史（跨batch）
            self.global_embedding_sum += current.sum(dim=0)  # [num_labels, hidden]
            self.global_step_count += 1
            del self._pending_update


class ContrastModel(BertPreTrainedModel):
    def __init__(self, config, batch_size=1, cls_loss=True, contrast_loss=True, contrast_mode='label_aware', graph=False, layer=1, data_path=None,
                 multi_label=False, lamb=1, lamb_1=0.1, threshold=0.01, tau=1, device="cuda", head=4, label_cpt=None, label_depths=None, label_dict=None, label_aware_embedding=None,
                 is_decoder=False, softmax_entropy=False, do_simple_label_contrastive=False, do_weighted_label_contrastive=False,
                 new_label_dict=None):
        super(ContrastModel, self).__init__(config)
        self.num_labels = config.num_labels
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.is_decoder = is_decoder
        self.softmax_entropy = softmax_entropy

        
        # TODO modify the classifier for flat multi label or sequence multi label
        with open(os.path.join(data_path, 'section_dict.pkl'), 'rb') as f:
            section_dict = pickle.load(f)
        self.section_dict = section_dict
        self.classifier1 = {}
        self.classifier2 = {}
        for k, v in section_dict.items():
            self.classifier1[k] = nn.Linear(config.hidden_size * len(v), config.hidden_size)
            self.classifier2[k] = (nn.Linear(config.hidden_size, len(v)))

        self.bert = BertModel(config)
        self.pooler = BertPoolingLayer(config, 'cls')
        self.batch_size = batch_size

        self.fc = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.fc1 = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.fc2 = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.gelu = nn.GELU()

        self.cls_loss = cls_loss
        self.contrast_loss = contrast_loss
        self.contrast_mode = contrast_mode

        if self.contrast_mode == 'straight_through':
            self.straight_fc = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.label_aware = LabelAware(config.hidden_size, config.hidden_size, head=head)

        self.graph_encoder = GraphEncoder(config, graph, layer=layer, data_path=data_path, threshold=threshold, tau=tau, label_dict=new_label_dict)
        hiera, _label_dict, r_hiera, label_depth = get_hierarchy_info(label_cpt)

        if not 'bgc' in data_path:
            with open(os.path.join(data_path, 'new_label_dict.pkl'), 'rb') as f:
                label_dict = pickle.load(f)
        else:
            label_dict = new_label_dict

        if ('rcv' in data_path) or ('bgc' in data_path):
            label_depth = [label_depth[k] for k, v in _label_dict.items()]
            label_dict = _label_dict
        else:
            label_depth = [label_depth[k] for k, v in label_dict.items()]
        label_depth = np.array(label_depth, dtype=np.int32)

        self.data_path = data_path

        self.hiera = hiera
        self.r_hiera = r_hiera
        self.label_depth = label_depth
        self.label_dict = label_dict
        self.r_label_dict = {v: k for k, v in label_dict.items()}

        if self.contrast_mode == 'attentive' or self.contrast_mode == 'simple_contrastive':
            self.contrast_proj = nn.Linear(config.hidden_size * 2, config.hidden_size, bias=True)
        self.lamb = lamb
        self.lamb_1 = lamb_1

        self.do_simple_label_contrastive = do_simple_label_contrastive
        self.do_weighted_label_contrastive = do_weighted_label_contrastive

        self.init_weights()
        self.multi_label = multi_label
        self.tau = tau  # temperature
        self.tanh = nn.Tanh()

        self.hidden_size = config.hidden_size

        if ('rcv' in data_path):
            self.label_path = {k: self.get_path(k) for k, v in _label_dict.items()}
        elif ('bgc' in data_path):
            self.label_path = {v: self.get_path(k) for k, v in _label_dict.items()}
        else:
            self.label_path = {v: self.get_path(k) for k, v in label_dict.items()}
        depth_label_path = {}
        for label in self.label_path:
            depth = len(self.label_path[label])
            if depth not in depth_label_path:
                depth_label_path[depth] = {}
            depth_label_path[depth][label] = self.label_path[label]
        self.depth_label_path = depth_label_path
        
    def get_label_path(self, labels, max_width=4):
        # labels: batch_size * num_labels
        # return: batch_size * max_width
        
        # max_width equals to the number of 1 in self.label_depth
        max_width = np.sum([1 for k, v in enumerate(list(self.label_depth)) if v == 1])
        batch_size = labels.shape[0]

        # fill the label_leaf by -1
        label_leaf = torch.ones((batch_size, max_width), dtype=torch.int64) * -1
        for i in range(batch_size):
            visited = set()
            j = 0
            label_depth = [(self.label_dict[idx],  self.label_depth[idx], idx) for idx, is_label in enumerate(labels[i]) if is_label == 1]
            sorted_label = sorted(label_depth, key=lambda x: x[1], reverse=True)
            for label, depth, idx in sorted_label:
                # check if sorted_label is all in the visited
                if j == max_width:
                    break # TODO fix the bug about j > max_width
                if (set(sorted_label).issubset(visited)) and (visited.issubset(set(sorted_label))):
                    break
                if label in visited:
                    continue
                else:
                    label_leaf[i][j] = idx
                    visited.add(label)
                    j += 1

                while (self.r_hiera[label] not in visited):
                    visited.add(self.r_hiera[label])
        return label_leaf

    def sample_path(self, leaf):
        # leaf: max_width
        # return: (max_width * 2),(max_width * 2)
        max_width = leaf.shape[0]
        path = torch.zeros((max_width * 2), dtype=torch.int64)
        gold = torch.zeros((max_width * 2), dtype=torch.int64)
        for i in range(max_width):
            if (leaf[i] == -1) or (self.label_depth[leaf[i]] <= 2):
                path[2 * i] = -1
                path[2 * i + 1] = -1
                gold[2 * i] = 0
                gold[2 * i + 1] = 0
            else:
                path[2 * i] = leaf[i]
                gold[2 * i] = 1
                
                same_path_label = [k for k, v in enumerate(list(self.label_depth)) if (v == self.label_depth[leaf[i]]) and (k != leaf[i])]
                path[2 * i + 1] = random.choice(same_path_label)
                gold[2 * i + 1] = 0

        return path, gold

    def get_label_path_embedding(self, leaf_idx, label_aware_embedding):
        # leaf_idx: batch_size * (max_width * 2)
        # label_aware_embedding: batch_size * num_labels * bert_hidden_size
        # return: batch_size * (max_width * 2)
        batch_size, max_width = leaf_idx.shape
        label_path_embedding = torch.zeros((batch_size, max_width), dtype=torch.float32)

        for i in range(batch_size):
            for j in range(max_width):
                if leaf_idx[i][j] == -1:
                    continue
                else:
                    path_embedding = label_aware_embedding[i][leaf_idx[i][j].item()]
                    label = self.label_dict[leaf_idx[i][j].item()]
                    count = 1
                    while label != 'Root':
                        path_embedding += label_aware_embedding[i][self.r_label_dict[label]]
                        label = self.r_hiera[label]
                        count += 1

                    label_path_embedding[i][j] = self.tanh(self.path_proj(torch.div(path_embedding, count))) # 
        return label_path_embedding 
    
    def get_leaf(self, labels):
        leaf = set()
        for label in labels:
            label = label[0]
            leaf = leaf - set(self.label_path[label.item()])
            leaf.add(label)
        return list(leaf)
    
    def get_path(self, label):
        path = []
        # label_name = label_dict[label]
        while label != 'Root':
            path.insert(0, label)
            label = self.r_hiera[label]
        return path
    
    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        # input_ids: batch_size * seq_len
        # labels: batch_size * num_labels
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        contrast_mask = None

        # sent_inputs: batch_size * seq_len * bert_hidden_size
        bert_out = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            embedding_weight=contrast_mask,
        )

        hidden_last, pooled, hidden_all = bert_out.last_hidden_state, bert_out.pooler_output, bert_out.hidden_states
        hidden_cls, encode_out = hidden_last[:, 0, :], hidden_last[:, 1:, :]   # encode_out是input_data
        # check with np.isclose
        
        # graph_inputs: 1 * label_num * bert_hidden_size
        graph_inputs = self.graph_encoder(lambda x: self.bert.embeddings(x)[0])

        # print("graph_inputs size:", graph_inputs.size())+
        # print("encode_out size:", encode_out.size())

        # repeat the graph_inputs to match the batch size
        graph_inputs = graph_inputs.repeat(encode_out.shape[0], 1, 1) # [batch_size, num_of_labels, hidden_size]

        sent_inputs_mask = attention_mask[:, 1:]
        
        # label_aware_embedding: batch_size * num_labels * hidden_size
        label_aware_embedding, attns = self.label_aware(encode_out, graph_inputs, sent_inputs_mask)

        # attns: batch_size * num_labels * seq_len

        loss = 0

        # fc label_aware_embedding: batch_size * num_labels * hidden_size
        # proj_label_embedding = self.fc(label_aware_embedding)
        proj_label_embedding = self.dropout(label_aware_embedding)

        if self.contrast_loss:
            if self.contrast_mode == 'label_aware':
                features = [proj_label_embedding[i, j] for i in range(proj_label_embedding.shape[0]) for j in range(self.num_labels) if labels[i, j]]
                features = torch.stack(features).to(self.device) # (batch_size * num_labels, hidden_size)
                features = torch.unsqueeze(features, 1) # (batch_size * num_labels, 1, hidden_size)
            elif self.contrast_mode == 'fusion':
                features = [torch.concat([proj_label_embedding[i, j], graph_inputs[0, j, :].squeeze()], dim=-1) for i in range(proj_label_embedding.shape[0]) for j in range(self.num_labels) if labels[i, j]]
                features = torch.stack(features).to(self.device) # (batch_size * num_labels, hidden_size + bert_hidden_size)
                features = torch.unsqueeze(features, 1) # (batch_size * num_labels, 1, hidden_size + bert_hidden_size)
            elif self.contrast_mode == 'attentive':
                # shape of proj_label_embedding: batch_size * num_labels * hidden_size

                # append the label embedding to the end of the sentence embedding
                # shape of proj_label_embedding: batch_size * num_labels * (hidden_size * 2)
                fusion_label_embedding = torch.cat([proj_label_embedding, graph_inputs], dim=-1) # batch_size * num_labels * (hidden_size * 2)
                fusion_attn_weights = self.contrast_proj(fusion_label_embedding) # batch_size * num_labels * hidden_size
                fusion_attn_weights = torch.softmax(fusion_attn_weights, dim=-1) # batch_size * num_labels * hidden_size
                fusion_attn_weights = torch.bmm(fusion_attn_weights, encode_out.transpose(1, 2)) # batch_size * num_labels * seq_len
                label_specifc_embedding = torch.bmm(fusion_attn_weights, encode_out) # batch_size * num_labels * bert_hidden_size

                features = label_specifc_embedding

            elif self.contrast_mode == 'simple_contrastive':
                fusion_label_embedding = torch.cat([proj_label_embedding, graph_inputs], dim=-1) # batch_size * num_labels * (hidden_size * 2)
                fusion_attn_weights = self.contrast_proj(fusion_label_embedding) # batch_size * num_labels * hidden_size
                fusion_attn_weights = torch.softmax(fusion_attn_weights, dim=-1) # batch_size * num_labels * hidden_size
                fusion_attn_weights = torch.bmm(fusion_attn_weights, encode_out.transpose(1, 2)) # batch_size * num_labels * seq_len

                label_specifc_embedding = torch.bmm(fusion_attn_weights, encode_out) # batch_size * num_labels * bert_hidden_size

                mask = labels.to(torch.bool)
                mask = mask.unsqueeze(-1).expand_as(label_specifc_embedding) # batch_size * num_labels * bert_hidden_size
                label_specifc_embedding = torch.masked_select(label_specifc_embedding, mask).view(-1, label_specifc_embedding.shape[-1]) # (batch_size * num_labels) * bert_hidden_size

                features = label_specifc_embedding
            elif self.contrast_mode == 'straight_through':
                features = proj_label_embedding  # batch_size * num_labels * hidden_size

        label_aware_embedding = self.dropout(features)

        if self.contrast_mode == 'straight_through':
            label_aware_embedding = features

        if not self.is_decoder:
        # flatten the label_aware_embedding into batch_size * (num_labels * hidden_size)
            # if self.classifier1 is not list:
            if not isinstance(self.classifier1, dict):
                cls_embedding = label_aware_embedding.view(-1, label_aware_embedding.shape[1] * label_aware_embedding.shape[2])
                # label_aware_embedding = self.dropout(label_aware_embedding)
                ## TODO try to add dropout here
                intermediate_embedding = self.classifier1(cls_embedding)
                intermediate_embedding = torch.relu(intermediate_embedding)
                logits = self.classifier2(intermediate_embedding)
            else:
                logits = np.zeros((label_aware_embedding.shape[0], self.num_labels))
                for k, v in self.section_dict.items():
                    # index the v in label_aware_embedding
                    label_aware_embedding_section = label_aware_embedding[:, v, :]
                    label_aware_embedding_section = label_aware_embedding_section.view(-1, label_aware_embedding_section.shape[1] * label_aware_embedding_section.shape[2])
                    # label_aware_embedding_section = self.dropout(label_aware_embedding_section)
                    intermediate_embedding = self.classifier1[k](label_aware_embedding_section)
                    intermediate_embedding = torch.relu(intermediate_embedding)
                    logits_section = self.classifier2[k](intermediate_embedding)
                    logits[:, v] = logits_section

        else:
        # Multi-label decoder (One for each layer)
            cls_embedding = label_aware_embedding
            logits = self.decoder(cls_embedding, cls_embedding) # batch_size * num_labels

        # only when training
        # labels.shape = [batch_size, num_labels]
        if labels is not None:
            if not self.multi_label:
                loss_fct = CrossEntropyLoss()
                target = labels.view(-1)
            elif self.softmax_entropy:
                target = labels.to(torch.float32)
            else:
                loss_fct = nn.BCEWithLogitsLoss()
                target = labels.to(torch.float32)  # batch_size * num_labels

            if self.softmax_entropy:
                # classification loss
                loss += multilabel_categorical_crossentropy(target, logits.view(-1, self.num_labels))
            else:
                loss += loss_fct(logits.view(-1, self.num_labels), target)


        if self.training:
            return {
                'loss': loss,
                'logits': logits,
                'labels': labels,
                'features': features if self.contrast_mode != 'straight_through' else self.straight_fc(features),
            }
        else:
            return {
                'loss': loss,
                'logits': logits,
                'features': features,
            }

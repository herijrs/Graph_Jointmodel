import torch
import torch.nn as nn
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertModel, BertConfig

from .torchcrf import CRF
from .module import IntentClassifier, SlotClassifier
from dgl.nn.pytorch import GATConv
import dgl

class JointBERT(BertPreTrainedModel):
    def __init__(self, config, args, intent_label_lst, slot_label_lst):
        super(JointBERT, self).__init__(config)
        self.args = args
        self.num_intent_labels = len(intent_label_lst)
        self.num_slot_labels = len(slot_label_lst)
        self.bert = BertModel(config=config)  # Load pretrained bert

        self.gat = GATConv(config.hidden_size, config.hidden_size, num_heads=4)

        # 定义门控单元的全连接层
        self.gate_layer = nn.Linear(config.hidden_size, config.hidden_size)

        self.intent_classifier = IntentClassifier(config.hidden_size, self.num_intent_labels, args.dropout_rate)
        self.slot_classifier = SlotClassifier(config.hidden_size, self.num_slot_labels, args.dropout_rate)

        if args.use_crf:
            self.crf = CRF(num_tags=self.num_slot_labels, batch_first=True)

    def forward(self, input_ids, attention_mask, token_type_ids, intent_label_ids, slot_labels_ids):


        outputs = self.bert(input_ids, attention_mask=attention_mask,
                            token_type_ids=token_type_ids)  # sequence_output, pooled_output, (hidden_states), (attentions)

        sequence_output = outputs[0]
        pooled_output = outputs[1]  # [CLS]


        graph = construct_graph(sequence_output)
        han_output = self.gat(graph, graph.ndata['feat'])
        semantic_output = torch.mean(han_output, dim=1).view(sequence_output.size(0),-1,768)

        # 添加门控单元
        gate_values = self.gate_function(sequence_output)  # 计算每个节点的门控值
        # gate_values = gate_values.unsqueeze(-1)  # 增加一个维度，用于与han_output相乘

        # 根据门控值对han_output进行加权融合，得到新的特征向量
        gated_han_output = semantic_output * gate_values

        # 更新slot_output特征向量
        slot_output = nn.LayerNorm(768)(sequence_output + gated_han_output)



        # Intent classification
        intent_logits = self.intent_classifier(torch.mean(semantic_output,dim=1))
        # Slot classification
        # slot_output = nn.LayerNorm(768)(sequence_output+semantic_output)
        slot_logits = self.slot_classifier(slot_output)


        total_loss = 0
        # 1. Intent Softmax
        if intent_label_ids is not None:
            if self.num_intent_labels == 1:
                intent_loss_fct = nn.MSELoss()
                intent_loss = intent_loss_fct(intent_logits.view(-1), intent_label_ids.view(-1))
            else:
                intent_loss_fct = nn.CrossEntropyLoss()
                intent_loss = intent_loss_fct(intent_logits.view(-1, self.num_intent_labels), intent_label_ids.view(-1))
            total_loss += intent_loss

        # 2. Slot Softmax
        if slot_labels_ids is not None:
            if self.args.use_crf:
                slot_loss = self.crf(slot_logits, slot_labels_ids, mask=attention_mask.byte(), reduction='mean')
                slot_loss = -1 * slot_loss  # negative log-likelihood
            else:
                slot_loss_fct = nn.CrossEntropyLoss(ignore_index=self.args.ignore_index)
                # Only keep active parts of the loss
                if attention_mask is not None:
                    active_loss = attention_mask.view(-1) == 1
                    active_logits = slot_logits.view(-1, self.num_slot_labels)[active_loss]
                    active_labels = slot_labels_ids.view(-1)[active_loss]
                    slot_loss = slot_loss_fct(active_logits, active_labels)
                else:
                    slot_loss = slot_loss_fct(slot_logits.view(-1, self.num_slot_labels), slot_labels_ids.view(-1))
            total_loss += self.args.slot_loss_coef * slot_loss

        outputs = ((intent_logits, slot_logits),) + outputs[2:]  # add hidden states and attention if they are here

        outputs = (total_loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions) # Logits is a tuple of intent and slot logits

    def gate_function(self, sequence_output):
        # 使用全连接层计算门控值
        gate_values = self.gate_layer(sequence_output)

        # 使用Sigmoid函数进行归一化，将门控值映射到0到1的范围
        gate_values = torch.sigmoid(gate_values)

        return gate_values


def construct_graph(bert_output):
    batch_size, seq_len, hidden_size = bert_output.size()
    # 构建边索引
    edge_index = []
    for i in range(batch_size):
        for j in range(seq_len - 1):
            src = i * seq_len + j
            dst = i * seq_len + j + 1
            edge_index.append((src, dst))
            edge_index.append((dst, src))

    # 创建DGL图对象
    graph = dgl.graph(edge_index, num_nodes=batch_size * seq_len)

    # 将BERT的输出向量作为节点特征
    graph.ndata['feat'] = bert_output.view(batch_size * seq_len, hidden_size)
    return graph


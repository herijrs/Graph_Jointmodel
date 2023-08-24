import torch.nn as nn


class IntentClassifier(nn.Module):  # intent分类的MLP全连接层
    def __init__(self, input_dim, num_intent_labels, dropout_rate=0.):
        super(IntentClassifier, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, num_intent_labels)

    def forward(self, x):
        # x:[batch_size,input_dim] 维度
        x = self.dropout(x)
        return self.linear(x)


class SlotClassifier(nn.Module):  # slot分类的MLP全连接层
    def __init__(self, input_dim, num_slot_labels, dropout_rate=0.):
        super(SlotClassifier, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, num_slot_labels)

    def forward(self, x):
        # x:[batch_size,max_seq_len,input_dim]维度
        x = self.dropout(x)
        return self.linear(x)

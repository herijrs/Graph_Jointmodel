import os

def vocab_process(data_dir):
    """
    Args:
       data_dir: 数据集所在的路径
    Returns:
       None
    Result:
       intent的label类型写入一个txt文件
       slot的label类型写入一个txt文件

    """
    # 标签集合输入到如下文件中
    slot_label_vocab = 'slot_label.txt'
    intent_label_vocab = 'intent_label.txt'

    # 找到训练集数据的路径 进行拼接
    train_dir = os.path.join(data_dir, 'train')
    # 收集intent标签
    with open(os.path.join(train_dir, 'label'), 'r', encoding='utf-8') as f_r, open(os.path.join(data_dir, intent_label_vocab), 'w',
                                                                                    encoding='utf-8') as f_w:
        # 新建intent_vocab集合 提取所有出现的intent的label类型
        intent_vocab = set()
        for line in f_r:
            line = line.strip()
            intent_vocab.add(line)

        # 由于数据集已经划分完成，可能会出现验证集中存在而训练集中不存在的标签，以"UNK"来进行标记
        # 当读取到验证集，需要将未见过的intent标签标记为"UNK"
        additional_tokens = ["UNK"]
        for token in additional_tokens:
            f_w.write(token + '\n')
        # 将vocab以字典序进行排列 也可以自定义其他排列方式
        intent_vocab = sorted(list(intent_vocab))
        for intent in intent_vocab:
            f_w.write(intent + '\n')

    # 收集slot槽位标签
    with open(os.path.join(train_dir, 'seq.out'), 'r', encoding='utf-8') as f_r, open(os.path.join(data_dir, slot_label_vocab), 'w',
                                                                                      encoding='utf-8') as f_w:
        # 新建slot_vocab集合 提取所有出现的slot的label类型
        slot_vocab = set()
        # 一个label序列如下： O O O O O B-fromloc.city_name O B-toloc.city_name B-round_trip I-round_trip
        # 按照空格分割得到label序列
        for line in f_r:
            line = line.strip()
            slots = line.split()
            for slot in slots:
                slot_vocab.add(slot)  # 放到slot_vocab集合中
        # label是以BIO形式进行标记，先按BIO后面的实体类别字典序排列，再按照BIO顺序排列
        slot_vocab = sorted(list(slot_vocab), key=lambda x: (x[2:], x[:2]))
        # Write additional tokens 写入其他标签
        # "UNK"标签和上面相同，"PAD"表示被填充的部分的label
        # Write additional tokens
        additional_tokens = ["PAD", "UNK"]
        for token in additional_tokens:
            f_w.write(token + '\n')

        for slot in slot_vocab:
            f_w.write(slot + '\n')


if __name__ == "__main__":
    vocab_process('atis')
    vocab_process('snips')

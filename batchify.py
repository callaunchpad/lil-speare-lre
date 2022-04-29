import torch

def get_batch(x, vocab, device):
    go_x, x_eos, styles = [], [], []
    max_len = max([len(s['text']) for s in x])
    for s in x:
        text = s['text']
        style = s['style']

        s_idx = [vocab.word2idx[w] if w in vocab.word2idx else vocab.unk for w in text]
        padding = [vocab.pad] * (max_len - len(text))
        go_x.append([vocab.go] + s_idx + padding)
        x_eos.append(s_idx + [vocab.eos] + padding)

        styles.append([style])

    return torch.LongTensor(go_x).t().contiguous().to(device), \
           torch.LongTensor(x_eos).t().contiguous().to(device), \
           torch.LongTensor(styles).t().contiguous().to(device)  # time * batch

def get_batches(data, vocab, batch_size, device):
    order = range(len(data))
    z = sorted(zip(order, data), key=lambda i: len(i[1]['text']))
    order, data = zip(*z)

    batches = []
    i = 0
    while i < len(data):
        j = i
        while j < min(len(data), i+batch_size) and len(data[j]['text']) == len(data[i]['text']):
            j += 1
        batches.append(get_batch(data[i: j], vocab, device))
        i = j
    return batches, order

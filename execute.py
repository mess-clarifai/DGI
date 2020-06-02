import os
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
from tqdm import tqdm, trange

from models import DGI, LogReg
from utils import process


def main():

    saved_graph = os.path.join('assets', 'saved_graphs', 'best_dgi.pickle')
    saved_logreg = os.path.join('assets', 'saved_graphs', 'best_logreg.pickle')

    dataset = 'cora'

    # training params
    batch_size = 1
    nb_epochs = 10000
    patience = 25
    lr = 0.001
    l2_coef = 0.0
    drop_prob = 0.0
    hid_units = 512
    sparse = True
    nonlinearity = 'prelu' # special name to separate parameters

    adj, features, labels, idx_train, idx_test, idx_val = process.load_data(dataset)

    features, _ = process.preprocess_features(features)

    nb_nodes = features.shape[0]
    ft_size = features.shape[1]
    nb_classes = labels.shape[1]

    adj = process.normalize_adj(adj + sp.eye(adj.shape[0]))

    if sparse:
        adj = process.sparse_mx_to_torch_sparse_tensor(adj)
    else:
        adj = (adj + sp.eye(adj.shape[0])).todense()

    features = torch.FloatTensor(features[np.newaxis])
    if not sparse:
        adj = torch.FloatTensor(adj[np.newaxis])
    labels = torch.FloatTensor(labels[np.newaxis])
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    print("Training Nodes: {}, Testing Nodes: {}, Validation Nodes: {}".format(len(idx_train), len(idx_test), len(idx_val)))

    model = DGI(ft_size, hid_units, nonlinearity)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_coef)

    if torch.cuda.is_available():
        print('Using CUDA')
        model.cuda()
        features = features.cuda()
        if sparse:
            sp_adj = sp_adj.cuda()
        else:
            adj = adj.cuda()
        labels = labels.cuda()
        idx_train = idx_train.cuda()
        idx_val = idx_val.cuda()
        idx_test = idx_test.cuda()

    b_xent = nn.BCEWithLogitsLoss()
    xent = nn.CrossEntropyLoss()
    cant_wait = 0
    best = 1e9
    best_t = 0

    if not os.path.exists(saved_graph):
        pbar = trange(nb_epochs)
        for epoch in pbar:
            model.train()
            optimiser.zero_grad()

            idx = np.random.permutation(nb_nodes)
            shuf_fts = features[:, idx, :]

            lbl_1 = torch.ones(batch_size, nb_nodes)
            lbl_2 = torch.zeros(batch_size, nb_nodes)
            lbl = torch.cat((lbl_1, lbl_2), 1)

            if torch.cuda.is_available():
                shuf_fts = shuf_fts.cuda()
                lbl = lbl.cuda()

            logits = model(features, shuf_fts, adj, sparse, None, None, None)

            loss = b_xent(logits, lbl)

            pbar.desc = 'Loss: {:.4f}'.format(loss)

            if loss < best:
                best = loss
                best_t = epoch
                cnt_wait = 0
                torch.save(model.state_dict(), saved_graph)
            else:
                cant_wait += 1

            if cant_wait == patience:
                tqdm.write('Early stopping!')
                break

            loss.backward()
            optimiser.step()


    print('Loading {}th Epoch'.format(best_t) if best_t else 'Loading Existing Graph')
    model.load_state_dict(torch.load(saved_graph))

    embeds, _ = model.embed(features, adj, sparse, None)
    train_embs = embeds[0, idx_train]
    val_embs = embeds[0, idx_val]
    test_embs = embeds[0, idx_test]

    train_lbls = torch.argmax(labels[0, idx_train], dim=1)
    val_lbls = torch.argmax(labels[0, idx_val], dim=1)
    test_lbls = torch.argmax(labels[0, idx_test], dim=1)

    tot = torch.zeros(1)
    if torch.cuda.is_available():
        tot = tot.cuda()

    accs = []

    print("\nValidation:")
    pbar = trange(50)
    for _ in pbar:
        log = LogReg(hid_units, nb_classes)
        opt = torch.optim.Adam(log.parameters(), lr=0.01, weight_decay=0.0)

        pat_steps = 0
        best_acc = torch.zeros(1)
        if torch.cuda.is_available():
            log.cuda()
            best_acc = best_acc.cuda()
        for _ in range(100):
            log.train()
            opt.zero_grad()

            logits = log(train_embs)
            loss = xent(logits, train_lbls)

            loss.backward()
            opt.step()

        logits = log(test_embs)
        preds = torch.argmax(logits, dim=1)
        acc = torch.sum(preds == test_lbls).float() / test_lbls.shape[0]
        accs.append(acc * 100)
        pbar.desc = "Accuracy: {:.2f}%".format(100 * acc)
        tot += acc

    torch.save(log.state_dict(), saved_logreg)

    accs = torch.stack(accs)
    print('Average Accuracy: {:.2f}%'.format(accs.mean()))
    print('Standard Deviation: {:.3f}'.format(accs.std()))

    print("\nTesting")
    logits = log(val_embs)
    preds = torch.argmax(logits, dim=1)
    acc = torch.sum(preds == val_lbls).float() / val_lbls.shape[0]
    print("Accuracy: {:.2f}%".format(100 * acc))


if __name__ == '__main__':
    log = main()

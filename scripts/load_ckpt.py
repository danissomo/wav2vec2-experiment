import logging
import time
import os
import torch
from transformers import AutoFeatureExtractor, Wav2Vec2ForSequenceClassification, Wav2Vec2Config
import torch.optim as optim
from typing import Tuple


def save(base_path, model, opt, **kwargs):
    kwargs.update({
        'model_state_dict' : model.state_dict(),
        'opt_state_dict' : opt.state_dict()
    })       
    torch.save(kwargs, os.path.join(base_path, f'model_{time.time()}.pt'))


def load(base_path = 'chkpts') ->  Tuple[AutoFeatureExtractor, Wav2Vec2ForSequenceClassification, optim.Adam]:
    if not os.path.exists(base_path):
        os.mkdir(base_path)
    files = [f for f in os.listdir(base_path) if os.path.isfile(os.path.join(base_path, f)) and f.endswith('.pt')]
    if len(files) == 0:
        logging.info('loading model')
        model = Wav2Vec2ForSequenceClassification.from_pretrained("superb/wav2vec2-base-superb-ks", num_labels = 8, ignore_mismatched_sizes=True).to('cuda')
        optimizer = optim.Adam(model.parameters(), lr=1*10e-6)
        save(base_path, model, optimizer, metric=0, loss=0)
    max_metric_chkpt = None
    for f in os.listdir(base_path):
        if os.path.isfile(os.path.join(base_path, f)) and f.endswith('.pt'):
            chkpt = torch.load(os.path.join(base_path, f))
            if max_metric_chkpt is None or max_metric_chkpt['metric'] < chkpt['metric']:
                max_metric_chkpt = chkpt
                print(f'path {os.path.join(base_path, f)}; metric {chkpt["metric"]}')
                
    cfg = Wav2Vec2Config.from_pretrained("superb/wav2vec2-base-superb-ks",  num_labels = 8, ignore_mismatched_sizes=True)
    model = Wav2Vec2ForSequenceClassification(cfg)
    model.load_state_dict(state_dict= max_metric_chkpt['model_state_dict'])
    model = model.to('cuda')
    feature_extractor = AutoFeatureExtractor.from_pretrained("superb/wav2vec2-base-superb-ks")
    optimizer = optim.Adam(model.parameters(), lr=1*10-6)
    optimizer.load_state_dict(state_dict= max_metric_chkpt['opt_state_dict'])
    return feature_extractor, model, optimizer
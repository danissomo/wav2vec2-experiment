import time
import logging
import wandb
import torch
import torch.optim.lr_scheduler as lr_scheduler 
from torchmetrics.functional import f1_score
from ravdess_dataset import AudioSet
from load_ckpt import load, save


def main():
    feature_extractor, model, optimizer = load()
    logging.info('loading dataset')
    data = AudioSet('RAVDESS-emotions-speech-audio-only/Audio_Speech_Actors_01-24')
    logging.info('loaded')
    percent = 80
    batch_size = 10
    train_size = int(len(data)*percent/100)
    test_size = len(data) - train_size
    train_set, val_set = torch.utils.data.random_split(data, [train_size, test_size])
    logging.info(f'train len: {len(train_set)}, eval len: {len(val_set)}')
    trainloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, 100)
    max_metric = 0
    wandb.init('wav2vec')
    for epoch in range(200): 
        running_loss = 0.0
        epoch_loss_train = 0.0
        f1_epoch_train = 0.0
        batches_count_train = 0
        for i, data in enumerate(trainloader):
            batches_count_train += 1
            inputs, labels = data
            inputs = feature_extractor(inputs.detach().numpy(), return_tensors="pt", sampling_rate=16000)
            optimizer.zero_grad()
            model.train()
            inputs["labels"] = labels
            inputs.to('cuda')
            pred = model(**inputs)
            loss = pred.loss
            logits = pred.logits
            predicted_class_ids = torch.argmax(logits, dim=-1).detach().cpu()
            metric = f1_score(predicted_class_ids, labels, task='multiclass', num_classes=8)
            loss.backward()
            optimizer.step()
            running_loss += loss.detach().cpu().item()
            f1_epoch_train += metric.item()
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss} f1: {metric.item()}')
            running_loss = 0.0
            epoch_loss_train += loss.detach().cpu().item()
        f1_epoch_train/=batches_count_train
        epoch_loss_train /= batches_count_train
        print('-'*20)
        print(f'TRAIN EPOCH LOSS {epoch_loss_train}; EPOCH METRIC: {f1_epoch_train}')
        print('TESTING')
        wandb.log({'avg train loss' : epoch_loss_train, 'avg f1 train' : f1_epoch_train})
        f1_epoch_test = 0.0
        epoch_loss_test = 0.0
        running_loss = 0.0
        batches_count_test = 0
        with torch.no_grad():
            for i, data in enumerate(testloader):
                batches_count_test += 1
                inputs, labels = data
                inputs = feature_extractor(inputs.detach().numpy(), return_tensors="pt", sampling_rate=16000)
                optimizer.zero_grad()
                model.eval()
                inputs["labels"] = labels
                inputs.to('cuda')
                pred = model(**inputs)
                loss = pred.loss
                logits = pred.logits
                predicted_class_ids = torch.argmax(logits, dim=-1).detach().cpu()
                metric = f1_score(predicted_class_ids, labels, task='multiclass', num_classes=8)
                running_loss += loss.detach().cpu().item()
                f1_epoch_test += metric.item()
                print(f'test [{epoch + 1}, {i + 1:5d}] loss: {running_loss} f1: {metric.item()}')
                running_loss = 0.0
                epoch_loss_test += loss.detach().cpu().item()
        f1_epoch_test/=batches_count_test
        epoch_loss_test /= batches_count_test
        print(f'TEST LOSS: {epoch_loss_test} epoch test {f1_epoch_test}')
        after_lr = optimizer.param_groups[0]["lr"]
        wandb.log({'avg test loss' : epoch_loss_test, 'avg f1 test' : f1_epoch_test, 'lr' : after_lr})
        if max_metric < f1_epoch_test:
            max_metric = f1_epoch_test
            print('saving ckpt')
            save('chkpts', model, optimizer, metric = f1_epoch_test, loss=epoch_loss_test)
        scheduler.step()




if __name__ == '__main__':
    main()
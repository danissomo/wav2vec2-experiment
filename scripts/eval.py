import torch
import torchaudio
from load_ckpt import load
from torch.utils.data import DataLoader, Dataset
import os
class TestAudioSet(Dataset):
    def __init__(self, path = 'test_emotion_dataset/original') -> None:
        self.audio_res = []
        for f in os.listdir(path):
            if f.endswith('.wav'):
                self.audio_res.append(os.path.join(path, f))

    def __getitem__(self, index):
        audio_tensor, sr = torchaudio.load(self.audio_res[index])
        resampled = torchaudio.functional.resample(audio_tensor, sr, 16000)
        return resampled
    
    def __len__(self):
        return len(self.audio_res)

def main():
    feature_extractor, model, optim = load()
    data = TestAudioSet()
    solve = []
    with torch.no_grad():
        for i, audio_tensor in enumerate(data):
            inputs = feature_extractor(audio_tensor.detach().numpy(), return_tensors="pt", sampling_rate=16000)
            inputs.to('cuda')
            pred = model(**inputs)
            solve.append(pred.logits.detach().cpu().numpy()[0])
            print(pred.logits.detach().cpu().numpy().shape)

    with open('solve.txt', 'w') as fd:
        for i, (filepath, out) in enumerate(zip(data.audio_res, solve)):
            out_str = f'{os.path.basename(filepath)}|'
            for v in out:
                out_str = out_str + str(v) +'|'
            fd.write(out_str+'\n')
            
            
if __name__ == '__main__':
    main()
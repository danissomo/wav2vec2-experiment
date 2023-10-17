import torch
import torchaudio
import os
import re
from torch.utils.data import Dataset, DataLoader
from typing import List, Any, Tuple
ID2LABEL = {
    '01' : 'neutral',
    '02' : 'calm',
    '03' : 'happy',
    '04' : 'sad',
    '05' : 'angry',
    '06' : 'fearfull',
    '07' : 'disgust',
    '08' : 'srprised'
}
LABEL2ID = { v : k for k, v in ID2LABEL.items()}

class AudioSet(Dataset):
    def __init__(self, path, sample_rate = 16000) -> None:
        self.data, self.labels, orig_sr = self.load_raw_data(path)
        self.sample_waw(orig_sr, sample_rate)


    def __getitem__(self, index) -> Tuple[int, torch.Tensor] :
        return  self.data[index], self.labels[index]

    def __len__(self):
        return len(self.data)


    def load_raw_data(self, data_path) -> Tuple[List[int], List[torch.Tensor], int]:
        data = []
        labels = []
        sr = []
        for sub_dir, dirs, files in os.walk(data_path):
            for file in files:
                # print("file:", file)
                if file.endswith(".wav"):
                    # print(os.path.join(sub_dir, file))

                    # Determine full path
                    full_path = os.path.join(sub_dir, file)

                    # Determine the basename of the file
                    base_name = os.path.basename(file)
                    # print(type(base_name))
                    # print("Actual file name: ", base_name)

                    # Determine into which category a given file belongs
                    matchObj = re.match(r'(\d\d-)(\d\d-)(\d\d-)', base_name, re.M|re.I)

                    if matchObj:
                        # print("matchObj.group(3): ", matchObj.group(3))
                        # Remove trailing hyphen
                        match_group_three = matchObj.group(3)
                        match = match_group_three.rstrip("-")
                        # print(match)
                        audio_class = torch.tensor(int(match)-1)
                        labels.append(audio_class)
                        # Move to the appropriate category based on match
                        audio_file, audio_sr = torchaudio.load(full_path)
                        sr.append(audio_sr)
                        data.append(audio_file[0, :])
        return data, labels, sr


    def sample_waw(self, orig_sr, sample_rate):
        max_shape = 0
        for i, tsr in enumerate(self.data):
            self.data[i] =  torchaudio.functional.resample(tsr, orig_sr[i], sample_rate)
            max_shape = tsr.shape[0] if max_shape < tsr.shape[0] else max_shape
        
        for i, tsr in enumerate(self.data):
            buf = torch.zeros(max_shape)
            buf[:tsr.shape[0]] = tsr
            self.data[i] = buf
        
                
        

if __name__ == '__main__':
    ds = AudioSet('RAVDESS-emotions-speech-audio-only/Audio_Speech_Actors_01-24')
    print(ds.labels)
import os
import torch
import torchaudio
import numpy as np
from tqdm import tqdm
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB

# List your audio directories here
AUDIO_DIRS = ["data/public_datasets/Train/train", "data/public_datasets/audio", "data/public_datasets/respiratory_sound/respiratory_sound_database/Respiratory_Sound_Database/audio_and_txt_files"]
#AUDIO_DIRS = ["audio_and_txt_files"]
# declarations
window_size = 10
num_window = 10
min_file_duration = 15
Train_split = 0.75
Test_split = 0.15
Val_split = 0.1

# def get_window_timestamp_wo_df(file_duration, window_size=10, num_window=10, min_step=0.5, debug=False):
#     step_size=(file_duration-window_size)/(num_window - 1) 
#     valid_clips = []
#     start = 0
#     end = window_size
#     for i in range(num_window):
#         if start + window_size > file_duration: continue
#         valid_clips.append(int(start * 10 ** 3) / 10 ** 3)
#         start += step_size
#     if not len(valid_clips) == num_window: f"Expected {num_window} valid clips, but got {len(valid_clips)}"
    
#     return valid_clips     
    
mel_transform = MelSpectrogram(
    sample_rate=16000,
    #n_fft=400,
    n_fft=1024, #butterfly calculation
    hop_length=160,
    n_mels=128
)    
db_transform = AmplitudeToDB()

def load_and_convert_ESC(file_path):
    try:
        waveform, sr = torchaudio.load(file_path)
    except Exception as e:
        print(f"Skipping {file_path}: {e}")
        return None

    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    mel = mel_transform(waveform)
    mel_db = db_transform(mel)
    
    mel_db = mel_db[:, :, :1024] if mel_db.shape[-1] > 1024 else \
            torch.nn.functional.pad(mel_db, (0, 1024 - mel_db.shape[-1]))
    return mel_db  # [1, 128, 1024]

def load_and_convert_other(file_path):
    try:
        waveform, sr = torchaudio.load(file_path)
    except Exception as e:
        print(f"Skipping {file_path}: {e}")
        return None

    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    
    mel = mel_transform(waveform)
    mel_db = db_transform(mel)    
    
    return mel_db 

def process_all():
    split = ['Train', 'Test', 'Val']
    for s in split: 
        #create directory
        base = "data/dia_tmp"
        other_datapth = os.path.join(base, s, "other_data")
        resp_datapth = os.path.join(base, s, "resp_data")
        good_datapth = os.path.join(base, s, "good")
        bad_datapth = os.path.join(base, s, "bad")
        if not os.path.exists(other_datapth) : os.makedirs(other_datapth)
        if not os.path.exists(resp_datapth) : os.makedirs(resp_datapth)
        if not os.path.exists(good_datapth) : os.makedirs(good_datapth)
        if not os.path.exists(bad_datapth) : os.makedirs(bad_datapth)
        for audio_dir in AUDIO_DIRS:
            print(audio_dir)
            assert Train_split + Test_split + Val_split == 1
            files = os.listdir(audio_dir)
            if s == 'Train': files = files[:int(Train_split*len(files))]
            if s == 'Test': files = files[int(Train_split*len(files)):int((1-Test_split)*len(files))]
            if s == 'Val': files = files[int((1-Test_split)*len(files)):]
            for file_name in files: 
                if not file_name.endswith('.wav'):
                    continue
                full_path = os.path.join(audio_dir, file_name)
                if not audio_dir == "data/public_datasets/respiratory_sound/respiratory_sound_database/Respiratory_Sound_Database/audio_and_txt_files": 
                    #print("going through here")
                    mel = load_and_convert_ESC(full_path)
                    if mel is None:
                            print("mel is None")
                            continue  # skip bad file
                    mel = mel.permute(0, 2, 1)  # -> [1, 1024, 128]
                    SAVE_DIR = other_datapth
                    save_name = f"{os.path.basename(audio_dir)}_{file_name.replace('.wav', '.pt')}"
                    torch.save(mel, os.path.join(SAVE_DIR, save_name))
                    print(f"Saved {SAVE_DIR, save_name}")

                else:       
                    mel = load_and_convert_other(full_path)
                    if mel is None:
                        print("mel is None")
                        continue  # skip bad file
                    num_mels = mel.shape[2]
                    print(num_mels)
                    if num_mels < 1024 : 
                        continue
                    elif num_mels >= 1500: 
                        num_iter = 10
                        step_size = int((num_mels-1024)/9)
                        assert step_size*9 + 1024 <= num_mels
                    else:
                        num_iter = 1
                        step_size = 1
                    mel = mel.permute(0, 2, 1)  # -> [1, 1024, 128]
                    SAVE_DIR = resp_datapth
                    for segment_id in range(num_iter) :
                        save_name = f"{os.path.basename(audio_dir)}_clip{segment_id}_{file_name.replace('.wav', '.pt')}"
                        torch.save(mel[:,segment_id*step_size:segment_id*step_size+1024,:], os.path.join(SAVE_DIR, save_name))
                        print(f"Saved {SAVE_DIR}/{save_name}")

if __name__ == "__main__":
    process_all()
    print("All files processed and saved.")
import os
import librosa
import numpy as np
import matplotlib.pyplot as plt

# load file
acc_audio_dir = r'E:\NRF_2022\MDPI\Extracted_Data\new_stft\acc_audio' # bgm audio folder path
file_list = os.listdir(acc_audio_dir)
print(f"audio num: {len(file_list)}")

# stft arg 
n_fft=2048
win_length = 2048
hop_length=512

for file in file_list:
    name, _ = os.path.splitext(file)
    audio_file_name = acc_audio_dir+ '\\' + file 
    # print(audio_file_name) 
    y, sr = librosa.load(audio_file_name)
    stft_result = librosa.stft(y, n_fft=n_fft, win_length = win_length, hop_length=hop_length)
    mel_spec = librosa.feature.melspectrogram(S=stft_result, sr=sr, n_mels=128, hop_length=512, win_length=win_length)
    librosa.display.specshow(librosa.amplitude_to_db(mel_spec, 
                                                    # ref=0.00002
                                                    ref=np.max), 
                                sr=sr, hop_length = hop_length, y_axis='mel', x_axis='time', cmap = 'jet')
    # plt.colorbar(format='%2.0f dB') # showing right colorbar
    plt.axis('off') # axis remove
    # plt.show()
    img_save_path=r'E:\NRF_2022\MDPI\Extracted_Data\new_stft_2024' # folder dir to save img 
    
    fname=img_save_path+ '\\' + name + '.png'
    plt.savefig(fname=fname, bbox_inches='tight', pad_inches=0) # remove padding and save
    
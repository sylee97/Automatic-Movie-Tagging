## Code for Automatic Movie Tag Generation System

### common
[installataion]
pytorch-cuda==11.6
pytorch==1.13.1
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia
python -m ipykernel install --user --name auto_tagging --display-name auto_tagging

### module 1) genere classification 

* dataset consists of 972 movie trailer video files  
    - [Dropbox Link : spectrogram image files for extracted audio from trailer](https://www.dropbox.com/home/automatic_movie_tagging/dataset/genre/new2)
- 6 genres : Action, Comedy, Crime, Drama, Horror, Romance

### module 2) mood tag classification

* dataset consists of 972 movie trailer video files  
    - [Dropbox Link : spectrogram image files for extracted audio from trailer](https://www.dropbox.com/home/automatic_movie_tagging/dataset/tag)
- 20 tags : listed in tag_list_mood.csv

### module 3) audio analysis

* test.csv file include 100 audios with 7 major feature values(chroma_stft, rms, spectral_centroid, spectral_bandwidth, spectral_rolloff, zcr, mfcc) 
and 10 extracted tags(idx) from musicnn models
* train with random forest model then evaluate
- run code in google colab: movie100Audio_genre_classification.ipynb






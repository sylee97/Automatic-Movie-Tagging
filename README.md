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
* model files : (3 versions) ResNet34, MobileNetv2, VGG16
    - [Link](https://www.dropbox.com/scl/fo/nf1eu8ujms4p784k8r848/h?rlkey=82t2w34hw0ko0rh6eemeqq68a&dl=0)

### module 2) mood tag classification

* dataset consists of 780 movie trailer video files  
    - [Dropbox Link : spectrogram image files for extracted audio from trailer](https://www.dropbox.com/home/automatic_movie_tagging/dataset/tag)
- 20 tags : listed in tag_list_mood.csv
* model files : (2 versions) BCE Loss, ASL Loss
    - [Link](https://www.dropbox.com/scl/fo/xbxa1tp52eub0x1us8ee8/h?rlkey=c3dmety4wcnwqif0tcurmqnun&dl=0)

### module 3) audio analysis

* dataset extracted accompany audio 987 files
      - [Link] https://drive.google.com/drive/folders/1Sc4YUPtTIEK25cWh-pQwQe-61yhJhykJ?usp=sharing
* test.csv file include 100 audios with 7 major feature values(chroma_stft, rms, spectral_centroid, spectral_bandwidth, spectral_rolloff, zcr, mfcc) 
and 10 extracted tags(idx) from musicnn models
* train with random forest model then evaluate
- run code in google colab: movie100Audio_genre_classification.ipynb






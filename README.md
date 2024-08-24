Python 환경 기반의 음성인식 실습에서는 아래의 내용을 다루고 있습니다.  
- 음성 특징 추출 
- 전통적인 기계학습 기반 음성인식 모델
- 딥러닝 기반의 음성인식 모델 

# __240824 실습__ 
## __실습 (1) Whisper Inference__ 

이 실습은 위스퍼 오픈소스를 사용하여 음성을 전사하는 방법에 다루고 있습니다.
음성 전사 후 질문 생성 AI, 요약 AI 등을 결합하여 활용할 수 있으며, 음성 번역도 사용할 수 있습니다. 

위스퍼를 사용할 수 있는 방법은 크게 3가지입니다. 
1. API 사용 - 유료로 제공하고 있으며 로컬 환경이 준비되지 않아도 추론이 가능합니다.
     -  아래 2가지는 무료 오픈소스를 사용하는 방법이며, 이를 위해 로컬 장치가 필요합니다. 본 실습에서는 코랩 환경을 이용합니다. 
2. openai-whisper 라이브러리 사용 - 실습(1)
3. 허깅페이스에서 whisper 모델을 불러와 사용 - 실습(2) 

실습 데이터셋은 전과 마찬가지로 aihub의 [소음 환경 음성인식 데이터](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=568)를 사용하였습니다. 아래 코드 상에 허깅페이스의 datasets 라이브러리로 로드를 하고 있는데, 해당 데이터셋이 삭제되었을 경우 직접 다운로드 받은 후 preprocess를 진행해야 합니다. 데이터 전처리는 aihub_preprocess.ipynb 파일을 참조해주세요. 

> __😃 실습 코드__   
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://drive.google.com/file/d/1ezz2HKxvjhTGBy_5Uw3KjfZcPAHIm1s-/view?usp=sharing)
>

## __실습 (2) Whisper Fine-tuning__   

> __😃 실습 코드__      
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://drive.google.com/file/d/1R_-7OkXBFwLkm0mXp-2kgZSvN4JB713n/view?usp=sharing)    
> 


# __240817 실습__
## __실습 (1) 음성파일 불러오기 및 STFT, MFCC__
 
 이번 실습에서는 Python 환경에서 음성 데이터를 읽어오고, 특징을 추출하는 방법에 대해 실습할 수 있습니다.  
 1. 데이터 불러오기 
 2. Spectrogram 특징 추출 
      - 신호 Framing 및 Windowing 
      - Short-time Fourier transform (STFT) 
 3. MFCC 특징 추출 
      - Mel Filter Bank 정의 및 연산  
      - librosa 라이브러리를 사용한 특징 추출

이 실습에서는 AIHub에서 제공하는 데이터를 사용합니다. 전체 데이터는 [소음 환경 음성인식 데이터](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=568)에서 다운로드 받을 수 있습니다.  '\*_SN.WAV' 파일은 음성과 소음이 섞여 있는 소리 데이터이며, '\*_SD.WAV' 파일은 깨끗한 화자 음성 소리 데이터입니다. 두 파일을 통해 소음 환경에 따른 음성 특징 추출 결과를 비교해 볼 수 있습니다.

> __😃 실습 코드__   
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/158V4HD9ZT4zicjGNvLM14eUZg-C-TriD?usp=sharing)

> __⭐️ 정답코드__   
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/163Wi_0uhgDksn1pikqWq4ISy4MvYKqnp?usp=sharing)    

## __실습(2) GMM-HMM을 이용한 음소 예측__
이 실습에서는 TIMIT(TIMIT Acoustic-Phonetic Continuous Speech Corpus) 데이터셋으로 음소를 예측하는 모델을 학습 시킵니다. 630명의 화자가 한 문장을 읽는 녹음 파일로 구성되어 있습니다. 음성의 문장, 단어, 음소 수준의 전사와 함께 제공됩니다.  
1. 학습 데이터 전처리 
2. GMM-HMM 모델 로드 및 학습 
3. 테스트 데이터 전처리 
4. 모델 성능 테스트

> __😃 실습 코드__  
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://drive.google.com/file/d/1ZQoAQLiQaHGqMVvgpTt3Fu3KR5gbmf4_/view?usp=sharing)

> __⭐️ 정답코드__  
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/12v9f_7-FPJBJ_fx-oFphmDqQ7HJC1Dzl?usp=sharing)   
 
# __Google Colab 설명__
### 환경 세팅 
본 실습은 [Google Colab](https://colab.google/)을 통해 진행됩니다. Colab을 사용하기 위해 구글 계정이 필요합니다. 
1. 구글 로그인 
2. Google Colaboratory 노트북 생성 ([LINK](https://colab.google/))
3. __구글 드라이브 마운트__


### 리눅스 시스템 기초 
코랩 왼쪽의 폴더 아이콘을 클릭하면 시스템의 폴더 구조를 볼 수 있습니다. 
처음 코랩에 접속했을 때 사용자가 작업하게 되는 위치는 /content이며, 이 폴더를 작업 폴더라고 부릅니다. 
/content에서 작업을 하고, 파일을 저장하면 코랩 세션이 끊어질 경우에 폴더가 리셋됩니다. 즉, 저장한 파일이 사라지게 됩니다. 

코드로 작업한 파일을 저장하기 위해서는 구글 드라이브와 연동을 한 후, 저장할 파일을 '/content' 폴더가 아닌 구글 드라이브 폴더에 저장해야 합니다. 

아래와 같이 명령어를 입력하면 작업 위치가 '/content'에서 'content/drive/MyDrive'로 이동하게 됩니다. MyDrive 폴더는 자신의 구글 드라이브 폴더입니다. 드라이브 마운트를 하지 않으면 해당 폴더가 존재하지 않기 때문에 에러가 발생합니다. 
```bash
cd /content/drive/MyDrive
```
또는 파이썬 코드 상에서도 os 라이브러리를 통해 작업 위치를 변경할 수 있습니다. 
```python
import os 
os.chdir('/contnet/drive/MyDrive')
```
아래 명령어로 현재 작업 폴더 위치를 확인할 수 있습니다.
```bash
pwd
```
아래 명령어로  현재 작업 폴더에 들어있는 파일과 폴더들을 확인해볼 수 있습니다.  
```bash
ls
```

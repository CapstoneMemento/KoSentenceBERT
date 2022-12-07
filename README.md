# CapstoneMemento KoSentenceBERT repository
- KoSentenceBERT를 실행하기 위한 레포지토리
- fork한 레포지토리: https://github.com/BM-K/KoSentenceBERT-SKT

## Installation
- 실행환경: 윈도우 10 Home
- 설치과정에서 에러가 날 수 있어 1. Ubuntu를 사용하는 방법과 2. 도커를 사용하는 방법 두 가지 설명드리겠습니다.
### 1. Ubuntu 환경에서의 실행
#### 0) Prerequisites
- Windows 터미널이 설치되어 있어야 합니다.
- python 3.9가 설치되어 있어야 합니다.
#### 1) WSL 설치하기
- Windows Powershell을 실행하여 다음 명령어를 실행합니다.
```
wsl --install
```

#### 2) Ubuntu 사용하기
- 윈도우 터미널을 실행하여 Ubuntu를 실행합니다.

#### 3) Anaconda 설치하기
- Python 3.9 버전을 가진 Anaconda를 설치하기 위해 [Anaconda Old packages lists](https://docs.anaconda.com/anaconda/packages/oldpkglists/)에서 2021.11 Anaconda를 설치합니다.
- Anaconda 패키지가 있는 [이 주소](https://repo.anaconda.com/archive/)에서 깔고 싶은 버전을 찾아 오른쪽 키를 눌러 링크를 복사한 후, /opt 경로에 설치합니다.
- 윈도우 터미널을 권리자 권한으로 실행하여 Ubuntu를 열고, 다음 명령어를 입력합니다.
```
cd /opt
wget https://repo.anaconda.com/archive/Anaconda3-2021.11-Linux-x86_64.sh
sudo bash Anaconda3-2021.11-Linux-x86_64.sh
```

#### 4) Anaconda 실행하기
- 다음 명령어를 입력합니다.
```
source /opt/anaconda3/bin/activate
conda init
source ~/.bashrc
```

#### 5) KoSentenceBERT 설치하기
- home 경로에 KoSentenceBERT를 설치하기 위해 다음 명령어를 입력합니다.
```
cd ~/
git clone https://github.com/SKTBrain/KoBERT.git
cd KoBERT
pip install -r requirements.txt
pip install .
cd ..
git clone https://github.com/BM-K/KoSentenceBERT_SKTBERT.git
cd KoSentenceBERT_SKTBERT
pip install -r requirements.txt
```

#### 6) KoSentenceBERT_SKTBERT의 transformers, tokenizers, sentence_transformers 디렉토리 옮기기
- 해당 디렉토리를 /opt/anaconda3/python3.9/site-packages에 이동합니다.


### 2. 도커를 사용하여 실행
#### 0) Prerequisites
- VScode가 설치되어 있어야 합니다.
- Docker가 설치되어 있어야 합니다.
#### 1) Docker 이미지를 pull 하기
- 사용한 Docker image <br>
    - https://hub.docker.com/r/klbm126/kosbert_image/tags <br>
- VScode 터미널에서 다음 명령어를 실행합니다.
```
docker pull klbm126/kosbert_image:latest
```

#### 2)  Docker 컨테이너를 실행하기
- VScode 터미널에서 다음 명령어를 실행합니다.
```
docker run -it --name=kosbert_image klbm126/kosbert_image:latest
```
- 해당 명령어를 실행하면 kosbert_image라는 이름의 컨테이너가 실행됩니다.

#### 3) KoSentenceBERT 설치하기
- 다음 명령어를 차례대로 입력합니다.
```
git clone https://github.com/SKTBrain/KoBERT.git
cd KoBERT
pip install -r requirements.txt
pip install .
cd ..
git clone https://github.com/BM-K/KoSentenceBERT_SKTBERT.git
cd KoSentenceBERT_SKTBERT
pip install -r requirements.txt
```

#### 4) KoSentenceBERT_SKTBERT의 transformers, tokenizers, sentence_transformers 디렉토리 옮기기
- 해당 디렉토리를 /opt/conda/lib/python3.7/site-packages에 이동합니다.


## Train Models
 - 모델 학습을 원하시면 KoSentenceBERT 디렉토리 안에 KorNLUDatasets이 존재하여야 합니다. <br>
 - STS를 학습 시 모델 구조에 맞게 데이터를 수정하였으며, 데이터와 학습 방법은 아래와 같습니다 : <br><br>
KoSentenceBERT/KorNLUDatates/KorSTS/tune_test.tsv <br>
<img src="https://user-images.githubusercontent.com/55969260/93304207-97afec00-f837-11ea-88a2-7256f2f1664e.png"></img><br>
*STS test 데이터셋의 일부* <br>
```
python training_nli.py      # NLI 데이터로만 학습
python training_sts.py      # STS 데이터로만 학습
python con_training_sts.py  # NLI 데이터로 학습 후 STS 데이터로 Fine-Tuning
```

## Pre-Trained Models
**pooling mode**는 **MEAN-strategy**를 사용하였으며, 학습시 모델은 output 디렉토리에 저장 됩니다. <br>
|디렉토리|학습방법|
|-----------|:----:|
|training_**nli**|Only Train NLI|
|training_**sts**|Only Train STS|
|training_**nli_sts**|STS + NLI|

학습된 pt 파일은 다음 드라이브에 있습니다. <br>
[[Google Drive Sharing]](https://drive.google.com/drive/folders/1fLYRi7W6J3rxt-KdGALBXMUS2W4Re7II?usp=sharing)<br>
<br>
<img src='https://user-images.githubusercontent.com/55969260/101112870-9cf42a00-3621-11eb-9bc0-788ba08638e1.png'><br>
각 폴더에 있는 result파일을 output 디렉토리에 넣으시면 됩니다. <br>
ex) sts 학습 파일 사용시 위 드라이브에서 sts/result.pt 파일을 output/training_sts/0_Transformer에 넣으시면 됩니다. <br>
output/training_sts/0_Transformer/result.pt <br>

## Performance
- Seed 고정, test set

|Model|Cosine Pearson|Cosine Spearman|Euclidean Pearson|Euclidean Spearman|Manhattan Pearson|Manhattan Spearman|Dot Pearson|Dot Spearman|
|:------------------------:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
|NLl|65.05|68.48|68.81|68.18|68.90|68.20|65.22|66.81|
|STS|**80.42**|**79.64**|**77.93**|77.43|**77.92**|77.44|**76.56**|**75.83**|
|STS + NLI|78.81|78.47|77.68|**77.78**|77.71|**77.83**|75.75|75.22|

- Performance comparison with other models [[KLUE-PLMs]](https://github.com/BM-K/Sentence-Embedding-is-all-you-need#performance).

## Application Examples
 - STS pretrained 모델을 통해 진행합니다.

### Semantic Compute
SemanticCompute.py는 corpus 문장과 queries 문장의 유사도를 일대일로 계산하고 출력하는 코드입니다.<br>
먼저 Corpus의 모든 문장에 대한 임베딩을 생성합니다.
```python
from sentence_transformers import SentenceTransformer, util
import numpy as np

# 판결 요지 하나를 문장 별로 비교한다.

model_path = './output/training_sts'

embedder = SentenceTransformer(model_path)

# Corpus with example sentences
corpus = [
    '특허권침해소송의 상대방이 제조 등을 하는 제품 또는 사용하는 방법(이하 ‘침해제품 등’이라고 한다)이 특허발명의 특허권을 침해한다고 하기 위해서는 특허발명의 특허청구범위에 기재된 각 구성요소와 그 구성요소 간의 유기적 결합관계가 침해제품 등에 그대로 포함되어 있어야 한다.',
    '침해제품 등에 특허발명의 특허청구범위에 기재된 구성 중 변경된 부분이 있는 경우에도, 특허발명과 과제 해결원리가 동일하고, 특허발명에서와 실질적으로 동일한 작용효과를 나타내며, 그와 같이 변경하는 것이 그 발명이 속하는 기술분야에서 통상의 지식을 가진 사람이라면 누구나 쉽게 생각해 낼 수 있는 정도라면, 특별한 사정이 없는 한 침해제품 등은 특허발명의 특허청구범위에 기재된 구성과 균등한 것으로서 여전히 특허발명의 특허권을 침해한다고 보아야 한다.']

corpus_embeddings = embedder.encode(corpus, convert_to_tensor=True)

# Query sentences:
queries = [
    '특허권침해소송의 상대방이 제조 등을 하는 제품이나 사용하는 방법(이하 ‘침해제품 등’)이 특허발명의 특허권을 침해한다 하기위해선 특허발명의 특허청구범위에 기재된 각 구성요소, 그리고 그 구성요소 간의 유기적 결합관계가 침해제품 등에 그대로 포함돼 있어야한다.',
    '침해제품 등에 특허발명의 특허청구범위에 기재된 구성 중 변경된 부분이 있는 경우 또한, 특허발명과 과제 해결원리가 동일하며 특허발명에서와 실질적으로 동일한 작용효과를 나타내고, 이처럼 변경하는 것이 해당 발명이 속하는 기술분야에서 통상의 지식을 가진 사람이라면 누구든지 쉽게 생각해낼수 있는 정도라면, 특별한 사정이 없는한 침해제품 등은 특허발명의 특허청구범위에 기재된 구성과 균등한 것으로 여전히 특허발명의 특허권을 침해한다고 봐야 된다.']

# 코사인 유사도에 기반하여 corpus[i]와 queries[i]의 유사도 구하기
idx = -1
for query in queries:
    idx = idx + 1
    query_embedding = embedder.encode(query, convert_to_tensor=True)
    cos_scores = util.pytorch_cos_sim(
        query_embedding, corpus_embeddings)[0][idx]
    cos_scores = cos_scores.cpu()

    # 사용자 입력 문장 출력
    print("\n\n======================\n\n")
    print("사용자:\n\n" + query)
    print("\n")

    # 퀴즈 정답과 계산된 유사도 출력
    print("정답:\n\n" + corpus[idx].strip(), "\n\n(Score: %.4f)" % (cos_scores))
    print("\n\n======================\n\n")

```
<br> 결과는 다음과 같습니다 :
```
======================


사용자:

특허권침해소송의 상대방이 제조 등을 하는 제품이나 사용하는 방법(이하 ‘침해제품 등’)이 특허발명의 특허권을 침해한다 하기위해선 특허발명의 특허청구범위에 기재된 각 구성요소, 그리고 그 구성요소 간의 유기적 결합관계가 침해제품 등에 그대로 포함돼 있어야한다.


정답:

특허권침해소송의 상대방이 제조 등을 하는 제품 또는 사용하는 방법(이하 ‘침해제품 등’이라고 한다)이 특허발명의 특허권을 침해한다고 하기 위해서는 특허발명의 특허청구범위에 기재된 각 구성요소와 그 구성요소 간의 유기적 결합관계가 침해제품 등에 그대로 포함되어 있어야 한다. 

(Score: 0.9882)


======================



======================


사용자:

침해제품 등에 특허발명의 특허청구범위에 기재된 구성 중 변경된 부분이 있는 경우 또한, 특허발명과 과제 해결원리가 동일하며 특허발명에서와 실질적으로 동일한 작용효과를 나타내고, 이처럼 변경하는 것이 해당 발명이 속하는 기술분야에서 통상의 지식을 가진 사람이라면 누구든지 쉽게 생각해낼수 있는 정도라면, 특별한 사정이 없는한 침해제품 등은 특허발명의 특허청구범위에 기재된 구성과 균등한 것으로 여전히 특허발명의 특허권을 침해한다고 봐야 된다.


정답:

침해제품 등에 특허발명의 특허청구범위에 기재된 구성 중 변경된 부분이 있는 경우에도, 특허발명과 과제 해결원리가 동일하고, 특허발명에서와 실질적으로 동일한 작용효과를 나타내며, 그와 같이 변경하는 것이 그 발명이 속하는 기술분야에서 통상의 지식을 가진 사람이라면 누구나 쉽게 생각해 낼 수 있는 정도라면, 특별한 사정이 없는 한 침해제품 등은 특허발명의 특허청구범위에 기재된 구성과 균등한 것으로서 여전히 특허발명의 특허권을 침해한다고 보아야 한다. 

(Score: 0.9846)


======================
```


## Reference
- [원본 레포](https://github.com/BM-K/KoSentenceBERT-SKT)
- [우분투 황경에서 KoSBERT 실행](https://seyeon.tistory.com/90)
- [도커를 사용하여 KoSBERT 실행](https://velog.io/@orange6/%ED%8C%90%EB%A1%80-%EC%95%94%EA%B8%B0-%EC%96%B4%ED%94%8C%EB%A6%AC%EC%BC%80%EC%9D%B4%EC%85%98-%EA%B0%9C%EB%B0%9C)

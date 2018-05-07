# HauntedTweet

트윗 아카이브를 받아 학습해, 주기적으로 사용자와 유사한 트윗을 작성합니다.

## 사용법

### Prerequisites

* Python 3
* TensorFlow 0.9
* KoNLPy
* jpype1

### 설치

### 테스트



## 설명

HauntedTweet은 인공신경망을 이용해 사용자의 트윗 아카이브를 학습시켜 사용자와 유사한 트윗을 작성합니다.

### 파일 구조

data는 사용자의 트윗 아카이브와 Parser가 위치한 폴더입니다.

dis는 주어진 텍스트와 사용자의 트윗의 유사도를 도출하는 코드가 위치한 폴더입니다.(*미구현)

gen은 사용자의 트윗과 유사한 텍스트를 생성하는 코드가 위치한 폴더입니다.

tests는 unittest를 이용한 테스트 코드 폴더입니다.

### 데이터 흐름


## 예시

 - 우선 다운로드 받은 트윗 아카이브 중 `tweets.csv` 파일을 `/data/` 폴더에 옮겨넣습니다. 형식을 유지한다면 일부는 편집해도 됩니다. 멘션, 링크, 사진, 알티한 글 등은 전처리 과정에서 전부 제외됩니다.

    python ./main.py -i

한국어 Parser를 이용해, 트윗을 token 단위로 분해하고 각 token의 빈도 분포와 학습 데이터를 저장합니다.

    python ./main.py -w 10000 -W word2vec_1.tfsav

Word2Vec 학습을 10,000 step만큼 (10,000개의 batch에 대해) 진행합니다. word2vec_1.tfsav(.data|.index|.meta) 파일이 없다면 처음부터 학습하여 결과를 저장하고, 있다면 중단한 곳에서부터 학습을 이어나갑니다.
	
    python ./main.py -w 0 -W word2vec_1.tfsav -g 10000 -G gen_1.tfsav

word2vec_1.tfsav에 저장되어 있는 Word2Vec 학습 데이터를 기반으로, Generator 학습을 10,000 step만큼 (10,000개의 batch에 대해) 진행합니다. gen_1.tfsav(.data|.index|.meta) 파일이 없다면 처음부터 학습하여 결과를 저장하고, 있다면 중단한 곳에서부터 학습을 이어나갑니다.

    python ./main.py -w 0 -W word2vec_1.tfsav -g 0 -G gen_1.tf sav -s 10000 -S result.txt

학습한 Generator를 이용하여, 100,00개의 문장을 생성해 main.py와 같은 디렉토리의 result.txt에 저장합니다. result.txt가 이미 있는 파일이라면 그 끝에 이어씁니다.

앞의 학습결과를 참고하기 위해, 0번 학습하는 편법을 사용한다는 점에 주의해주세요. 이 때문에 메모리 부족이 일어날 수 있습니다.


## Hyperparameter 조절

### Word2Vec

초기에 낮은 window_size, 높은 subsampling 계수 T와 학습계수를 사용하다가 점점 높였/낮췄습니다.

### Generator

초기에 낮은 batch_size와 sequence_length를 사용하다가, 높은 학습계수를 사용하다가 점점 높였/낮췄습니다.


## 코드 수정

코드에서 몇 가지 수정하고 싶을만한 부분을 소개합니다.

### main.py

`word2vec_batch_size, gen_batch_size, gen_seq_length`를 변경할 수 있습니다.

`generator.nn_init()`에 넘겨주는 argument 중 `learning_rate`를 변경할 수 있습니다.

`embedding_size, gen_hidden_size`를 변경하여 단어벡터의 사이즈, Generator RNN 모델의 레이어 수와 내부 size를 변경할 수 있습니다. 이 변수를 변경하면 변경 전의 학습데이터는 쓸 수 없게 됩니다.

### `word2vec.py`

`Word2Vec.generate_batch()`에서 `T`를 변경하여, subsampling 계수를 조절할 수 있습니다.

`Word2Vec.tf_init()`에서 학습과 관련된 대부분의 변수를 조절가능합니다.

`Word2Vec.give_code()` 정의에서 `max_word_count`를 변경할 수 있습니다. 이 변수를 변경하면 변경 전의 학습데이터는 쓸 수 없게 됩니다.

### `generator.py`

`Generator.nn_init()`에서, 모델에 쓰이는 RNNcell 종류를 변경할 수 있습니다. 기본값은 GRUCell(`tf.nn.rnn_cell.GRUCell`)입니다.

`Generator.batch_real_data()`에서, 지역변수 `input_length` 값을 변경하는 걸로 학습데이터에 변화를 줄 수 있습니다.

 - `input_length = len(input_tokens) - 2`를 사용하면, 문장을 끝내는 위치를 학습하지 않습니다.
 - `input_length = len(input_tokens) - 1`를 사용하면, 문장을 끝내는 위치를 학습합니다.
 - `input_length = len(input_tokens)`를 사용하면, 문장을 끝낸 뒤 다시 시작하도록 학습합니다. 프로그램이 한 문장 씩을 학습하도록 되어 있기 때문에, 추천하지 않습니다.

`#Randomly ignore sentence with UNK` 행 밑에, `if np.random.random() < 0.8:`라는 행이 있습니다. 이 값을 변경하는 걸 통해서, 매우 드문 단어가 들어가는 문장이라도 학습시킬지, 학습시키지 않을지, 확률적으로 일부만 학습시킬지 정할 수 있습니다. `generate()`에서 문장을 생성할 때 `UNK` 토큰을 기피하도록 되어 있다는 걸 참고해주세요.

neive하게 추측한 문장의 출현 확률을 통한 subsampling은 동작하지 않습니다. `pass`행을 `continue`로 바꾸는 걸 통해 다시 동작시킬 수 있습니다.

`Generator.train_real_data()`에서, `session.run()`에 전달하는 Dropout 확률의 보수인 `self.keep_input, self.keep_output, self.keep_state` 값을 조절할 수 있습니다. 기본값은 `self.keep_input : 1.0, self.keep_output : 0.5, self.keep_state : 1.0`입니다.

`Generator.generate()`에서, 문장 생성 알고리즘을 변경할 수 있습니다.

`initial_state`를 정의할 때, `tf.random_normal()`에 넘겨주는 `stddev` 값을 변경할 수 있습니다. 학습에 사용한 모델은 state가 zero state인 상태에서 `<go>` 토큰을 받는데, non-zero인 `stddev` 값을 사용하면 좀 더 다양한 상황에서의 문장 시작을 가정할 수 있습니다. 0을 사용하면 첫 단어의 확률 분포는 결정론적으로 정해집니다.

`#Only if UNK token is dominant, use UNK.` 행 밑에서만 `UNK` 토큰이 쓰일 수 있는 유일한 경우를 결정합니다. `UNK` 토큰의 출현확률이 0.99 이상일 때만, `UNK` 토큰을 사용합니다. 그 외의 경우에는 `UNK` 토큰을 제외하고 출현확률을 계산합니다.

`#Give probability weight` 행 밑에서, 좀 더 자연스러운 문장이 나올 수 있도록 확률을 조절합니다. `np.power(new_word_prob[j, :], 1.0)`에서 두번째 argument를 조절해서, 확률이 높은 단어의 우선도를 조절할 수 있습니다. 

 - `1.0`은 확률을 그대로 반영합니다.
 - `<1.0`인 값은 출현확률을 평탄하게 만듭니다. 문장이 무의미해질 가능성이 큽니다.
 - `>1.0`인 값은 출현확률을 과장시킵니다. 같은 단어·구절을 반복할 가능성이 큽니다.

그 다음 행은 `<eos>` 토큰의 출현확률을 조절합니다. 출현확률을 `4.0`으로 나누는데, 출현확률이 1에 가까운 경우에는 영향이 거의 없지만, 그렇지 않은 경우 출현확률이 빠르게 낮아집니다. 이 값은 조절할 수 있습니다. `1.0`을 쓰면 출현확률을 조절하지 않습니다.


### `unparser.py`

`Unparser.unparse()`는 신경망의 overfit에 대처하여, 문장을 덜 왜곡되어 보이게 하는 몇 가지 기능을 갖고 있습니다.

 - `<go>` 토큰, `<eos>` 토큰, `……` 토큰이 반복되는 경우, 반복을 무시하고 하나의 토큰으로 대체합니다.
 - 그 외의 단어가 반복되는 경우, 단어 뒤에 '*[N]'을 붙이는 걸로 대신합니다.
 
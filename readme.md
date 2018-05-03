# HauntedTweet

트윗 아카이브를 받아 학습해, 주기적으로 사용자와 유사한 트윗을 작성합니다.

## 사용법

### Prerequisites

* Python
* TensorFlow
* KoNLPy
* jpype1

### 설치

### 테스트



## 설명

HauntedTweet은 사용자의 트윗 아카이브로부터 GAN을 이용해 사용자와 유사한 트윗을 작성합니다.

### 파일 구조

data는 사용자의 트윗 아카이브와 Parser가 위치한 폴더입니다.

dis는 주어진 텍스트와 사용자의 트윗의 유사도를 도출하는 코드가 위치한 폴더입니다.

gen은 사용자의 트윗과 유사한 텍스트를 생성하는 코드가 위치한 폴더입니다.

tests는 unittest를 이용한 테스트 코드 폴더입니다.

### 데이터 흐름



## 예시




## Hyper parameter 조절

### Word2Vec

초기에 낮은 window_size, 높은 subsampling 계수 T와 학습계수를 사용하다가 점점 높였/낮췄다.

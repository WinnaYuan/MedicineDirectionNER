# This is a NER method based machine learning of BiLSTM + CRF

##环境
* 任何操作系统
* Tensorflow 1.10
* python 3

##要求
* numpy
* pickle


## 使用:


there are 3 modes to choose: train test and demo, use model to predict result, please run:

	python main.py --mode "demo"


if you want to use demo directly, there is an interface predictNER() in demo.py

	def predictNER(sents)

argument is a list as [direction1, direction2...],
return is a list as [[direction1, correspond tag1], [direction2, correspond tag2]...]





install:
	python -m pip install --upgrade pip && \
	pip install -r requirements.txt

lint:
	pylint --disable=R,C utils/*.py preprocessing/*.py

preprocess:
	python -m preprocessing.preprocess

segment:
	python -m preprocessing.segment

create_2d:
	python -m preprocessing.create_2d_data

train_svm:
	python -m train.svm.train

predict_svm:
	python -m predict.svm.predict "$(AUDIO)"
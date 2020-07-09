get-amazon:
	mkdir -p data/amazon
	python get_dataset.py --data_source amazon --data_dir ./data/amazon --verbose --download --train_bpe --collect_data

collect-amazon:
	python get_dataset.py --data_source amazon --data_dir ./data/amazon --verbose --collect_data

get-opensubtitles:
	mkdir -p data/opensubtitles
	python get_dataset.py --data_source opensubtitles --data_dir ./data/opensubtitles --verbose --download --train_bpe --collect_data --context_token \<CTX\>

collect-opensubtitles:
	python get_dataset.py --data_source opensubtitles --data_dir ./data/opensubtitles --verbose --collect_data --context_token \<CTX\>

install-apex:
	git clone https://github.com/NVIDIA/apex
	cd apex
	pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
	cd ../

install-requirements:
	pip install -r requirements.txt

train-amazon:
	python train.py --data_source amazon --data_dir ./data/amazon --checkpoint_path ./data/amazon/checkpoint --use_kl --gpus 4

train-opensubtitles:
	python train.py --data_source opensubtitles --data_dir ./data/opensubtitles --checkpoint_path ./data/opensubtitles/checkpoint --use_kl --gpus 4

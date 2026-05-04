# sh ./scripts/download/dowanload.sh

# download the up-to-date benchmarks and checkpoints
# provided by OpenOOD v1.5
python ./scripts/download/download.py \
	--contents 'datasets' 'checkpoints' \
	--datasets 'default' \
	--checkpoints 'default' \
	--save_dir './data' './results' \ba
	--dataset_mode 'benchmark'

task=SST-2
gpu=$1

for seed in 0 ; do
for train_seed in 0  ; do # 128 0 21 42 87 100
for model_type in bert-base-uncased ; do #
for n_label in 346; do # the number of training examples per class
# for prefix in  attrprompt ; do  # TODO: change to our name
for prefix in ourgen ; do

seed=${seed}
train_seed=${train_seed}
max_seq_len=128
max_seq_len_test=128

eval_batch_size=256
steps=100
#####################
gen_model=gpt3
gpt_model="gpt-3.5-turbo"
data_dir="../../datasets/sst-2/ourgen"   # TODO(rjy): need change
# data_dir = '/media/sdb1/nighoodRen/generate_data/AttrPrompt/datasets/sst-2/ourgen'
gen_model=${gen_model}_${prefix}_${n_label}
train_file="original_fewgen_all.jsonl"

lr=2e-5 # 2e-5
# lr = 5e-6
batch_size=32
epochs=30
weight_decay=1e-3

output_dir=result/${task}/model/original_fewgen_all
cache_dir=result/${task}/cache/original_fewgen_all
log_file=${output_dir}/output.log
mkdir -p ${output_dir}
mkdir -p ${cache_dir}

train_cmd="CUDA_VISIBLE_DEVICES=${gpu} python3 main.py --task=${task} \
	--train_file=${train_file} --dev_file=dev.jsonl --test_file=test.jsonl \
	--unlabel_file=unlabeled.json --tokenizer=${model_type} \
	--gen_model=${gen_model} --data_dir=${data_dir} --seed=${seed} --train_seed=${train_seed} \
	--cache_dir=${cache_dir} --output_dir=${output_dir}  \
	--gpu=${gpu} --num_train_epochs=${epochs} --weight_decay=${weight_decay} --learning_rate=${lr}  \
	--batch_size=${batch_size} --eval_batch_size=${eval_batch_size} \
	--max_seq_len=${max_seq_len} --max_seq_len_test=${max_seq_len_test} --auto_load=1 \
	--max_steps=${steps} --model_type=${model_type}"
echo $train_cmd > $log_file 2>&1
eval $train_cmd >> $log_file 2>&1

done
done
done

done 
done

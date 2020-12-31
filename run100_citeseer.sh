mkdir citeseer
for num in $(seq 0 99)
do
	python train_grand_cite.py  --dataset citeseer --patience 100 --seed $num --dropnode_rate 0.5  --cuda_device 7 > citeseer/"$num".txt
done


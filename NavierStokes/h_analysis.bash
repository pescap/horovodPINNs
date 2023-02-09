epochs=30000;
lr=1e-4;
logfile=out.log

for N in 100 200 350 500 1000 2000 5000 10000 20000 40000 60000; do
for gpu in {0..7}; do  i=$(($gpu+1234));echo $i;CUDA_VISIBLE_DEVICES=${gpu} python hvd_hval.py --save --seed ${i} --N ${N} --lr ${lr} --epochs ${epochs}  2> /dev/null&echo "$N"-"$gpu"> "$logfile";
done
wait;
done
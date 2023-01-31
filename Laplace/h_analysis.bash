epochs=20000;
lr=1e-4;
logfile=out.log

for N in 8 16 32 64 80 100 105 110 128 150 200 256 512 1024 2048 4096 8192 16384 32768 65536 131072; do
for gpu in {0..7}; do  i=$(($gpu+1234));echo $i;CUDA_VISIBLE_DEVICES=${gpu} python hvd_hval.py --save --seed ${i} --N $N --lr ${lr} --epochs ${epochs}  2> /dev/null&echo "$N"-"$gpu"> "$logfile";
done
wait;
done
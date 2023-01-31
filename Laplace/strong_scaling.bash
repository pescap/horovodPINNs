epochs=20000;
lr=1e-4;
logfile=out.log

# Perform the weak scaling

np=2;
# 2 GPUs
CUDA_VISIBLE_DEVICES=0,1 horovodrun -np ${np} -H localhost:${np} python hvd_hval.py --N 128 --lr ${lr} --seed 1234 --epochs ${epochs} --save 2> /dev/null&
echo "$np"-1234> "$logfile";
CUDA_VISIBLE_DEVICES=2,3 horovodrun -np ${np} -H localhost:${np} python hvd_hval.py --N 128 --lr ${lr} --seed 1235 --epochs ${epochs} --save 2> /dev/null&
echo "$np"-1235> "$logfile"
CUDA_VISIBLE_DEVICES=4,5 horovodrun -np ${np} -H localhost:${np} python hvd_hval.py --N 128 --lr ${lr} --seed 1236 --epochs ${epochs} --save 2> /dev/null&
echo "$np"-1236> "$logfile";
CUDA_VISIBLE_DEVICES=6,7 horovodrun -np ${np} -H localhost:${np} python hvd_hval.py --N 128 --lr ${lr} --seed 1237 --epochs ${epochs} --save 2> /dev/null&
echo "$np"-1237> "$logfile";
wait;
CUDA_VISIBLE_DEVICES=0,1 horovodrun -np ${np} -H localhost:${np} python hvd_hval.py --N 128 --lr ${lr} --seed 1238 --epochs ${epochs} --save 2> /dev/null&
echo "$np"-1238> "$logfile";
CUDA_VISIBLE_DEVICES=2,3 horovodrun -np ${np} -H localhost:${np} python hvd_hval.py --N 128 --lr ${lr} --seed 1239 --epochs ${epochs} --save 2> /dev/null&
echo "$np"-1239> "$logfile";
CUDA_VISIBLE_DEVICES=4,5 horovodrun -np ${np} -H localhost:${np} python hvd_hval.py --N 128 --lr ${lr} --seed 1240 --epochs ${epochs} --save 2> /dev/null&
echo "$np"-1240> "$logfile";
CUDA_VISIBLE_DEVICES=6,7 horovodrun -np ${np} -H localhost:${np} python hvd_hval.py --N 128 --lr ${lr} --seed 1241 --epochs ${epochs} --save 2> /dev/null&
echo "$np"-1241> "$logfile";
wait;

np=4;
# 4 GPUs
CUDA_VISIBLE_DEVICES=0,1,2,3 horovodrun -np ${np} -H localhost:${np} python hvd_hval.py --N 64 --lr ${lr} --seed 1234 --epochs ${epochs} --save 2> /dev/null&
echo "$np"-1234> "$logfile";
CUDA_VISIBLE_DEVICES=4,5,6,7  horovodrun -np ${np} -H localhost:${np} python hvd_hval.py --N 64 --lr ${lr} --seed 1235 --epochs ${epochs} --save 2> /dev/null&
echo "$np"-1235> "$logfile";
wait;

CUDA_VISIBLE_DEVICES=0,1,2,3 horovodrun -np ${np} -H localhost:${np} python hvd_hval.py --N 64 --lr ${lr} --seed 1236 --epochs ${epochs} --save 2> /dev/null&
echo "$np"-1236> "$logfile";
CUDA_VISIBLE_DEVICES=4,5,6,7  horovodrun -np ${np} -H localhost:${np} python hvd_hval.py --N 64 --lr ${lr} --seed 1237 --epochs ${epochs} --save 2> /dev/null&
echo "$np"-1237> "$logfile";
wait;

CUDA_VISIBLE_DEVICES=0,1,2,3 horovodrun -np ${np} -H localhost:${np} python hvd_hval.py --N 64 --lr ${lr} --seed 1238 --epochs ${epochs} --save 2> /dev/null&
echo "$np"-1238> "$logfile";
CUDA_VISIBLE_DEVICES=4,5,6,7 horovodrun -np ${np} -H localhost:${np} python hvd_hval.py --N 64 --lr ${lr} --seed 1239 --epochs ${epochs} --save 2> /dev/null&
echo "$np"-1239> "$logfile";
wait;

CUDA_VISIBLE_DEVICES=0,1,2,3 horovodrun -np ${np} -H localhost:${np} python hvd_hval.py --N 64 --lr ${lr} --seed 1240 --epochs ${epochs} --save 2> /dev/null&
echo "$np"-1240> "$logfile";
CUDA_VISIBLE_DEVICES=4,5,6,7 horovodrun -np ${np} -H localhost:${np} python hvd_hval.py --N 64 --lr ${lr} --seed 1241 --epochs ${epochs} --save 2> /dev/null&
echo "$np"-1241> "$logfile";
wait;

np=8;
# 8 GPUs
horovodrun -np ${np} -H localhost:${np} python hvd_hval.py --N 32 --lr ${lr} --seed 1234 --epochs ${epochs} --save 2> /dev/null;
echo "$np"-1234> "$logfile";
horovodrun -np ${np} -H localhost:${np} python hvd_hval.py --N 32 --lr ${lr} --seed 1235 --epochs ${epochs} --save 2> /dev/null;
echo "$np"-1235> "$logfile";
horovodrun -np ${np} -H localhost:${np} python hvd_hval.py --N 32 --lr ${lr} --seed 1236 --epochs ${epochs} --save 2> /dev/null;
echo "$np"-1236> "$logfile";
horovodrun -np ${np} -H localhost:${np} python hvd_hval.py --N 32 --lr ${lr} --seed 1237 --epochs ${epochs} --save 2> /dev/null;
echo "$np"-1237> "$logfile";
horovodrun -np ${np} -H localhost:${np} python hvd_hval.py --N 32 --lr ${lr} --seed 1238 --epochs ${epochs} --save 2> /dev/null;
echo "$np"-1238> "$logfile";
horovodrun -np ${np} -H localhost:${np} python hvd_hval.py --N 32 --lr ${lr} --seed 1239 --epochs ${epochs} --save 2> /dev/null;
echo "$np"-1239> "$logfile";
horovodrun -np ${np} -H localhost:${np} python hvd_hval.py --N 32 --lr ${lr} --seed 1240 --epochs ${epochs} --save 2> /dev/null;
echo "$np"-1240> "$logfile";
horovodrun -np ${np} -H localhost:${np} python hvd_hval.py --N 32 --lr ${lr} --seed 1241 --epochs ${epochs} --save 2> /dev/null;
echo "$np"-1241> "$logfile";
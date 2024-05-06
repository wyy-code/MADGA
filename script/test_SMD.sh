for file in /data/user/wyy/Data/input/processed/machine*train*
do 
    var=${file##*/}
    # echo $var
    echo ${var%_*}
    CUDA_VISIBLE_DEVICES=1 python test.py --name=${var%_*} --n_blocks 2 
    wait
done

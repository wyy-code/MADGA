for file in /data/user/wyy/MTGFLOW-main/Data/input/processed/machine*train*
do 
    var=${file##*/}
    # echo $var
    echo ${var%_*}
    CUDA_VISIBLE_DEVICES=2 nohup python -u main.py\
        --n_blocks=2\
        --batch_size=256\
        --window_size=60\
        --train_split=0.6\
        --name=${var%_*}\
        > OCCmain${var%_*}.log 2>&1 &
    wait
done

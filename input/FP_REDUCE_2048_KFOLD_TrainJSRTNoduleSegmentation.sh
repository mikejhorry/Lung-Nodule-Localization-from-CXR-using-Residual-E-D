export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/mjhorry/anaconda3/lib


# first 10
dim=2048

for test in JSRT_SOURCE_HE_SEG_A
do
for model in RESUNET-M RESUNET-L 
do
  for lr in 1e-5
  do 
    for k in {0..90..3}
    #for k in {48..90..3}
      do
         python "JSRT_Nodule_Chest_Xray_kfold_train_test.py" -t $test -m $model -lr $lr -bs 1 -d "/data/mjhorry/Experiments/JSRT Nodule Segmentation/input/nodule_files/$test/crossval" -e 25 -a true -x $dim -k $k -in True -inf True -best True
      done 
  done
done
done


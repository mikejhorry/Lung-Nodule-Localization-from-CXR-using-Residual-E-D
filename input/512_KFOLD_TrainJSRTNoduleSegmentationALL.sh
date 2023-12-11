export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/mjhorry/anaconda3/lib


# first 10
dim=512

#for test in JSRT_SOURCE_A JSRT_SOURCE_SEG_A JSRT_SOURCE_HE_A JSRT_SOURCE_HE_SEG_A
#for test in JSRT_SOURCE_STD_A JSRT_SOURCE_STD_SEG_A
for test in JSRT_SOURCE_SUP_STD_A JSRT_SOURCE_SUP_STD_SEG_A
do
for model in RESUNET-S RESUNET-M RESUNET-L #UNET
do
  for lr in 1e-5
  do 
     python "JSRT_Nodule_Chest_Xray_kfold_train_test.py" -t $test -m $model -lr $lr -bs 1 -d "/data/mjhorry/Experiments/JSRT Nodule Segmentation/input/nodule_files/$test/crossval" -e 25 -a true -x $dim -k 0 -in True
  done
done
done


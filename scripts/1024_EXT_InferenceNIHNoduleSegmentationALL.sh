export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/mjhorry/anaconda3/lib

# first 10
dim=1024
k=0
lr=1e-5

for test in JSRT_SOURCE_SEG_A_EXT JSRT_SOURCE_SEG_B_EXT JSRT_SOURCE_SEG_C_EXT JSRT_SOURCE_SEG_D_EXT
do
 for model in RESUNET-S RESUNET-M RESUNET-L
 do 
# python "JSRT_Nodule_Chest_Xray_kfold_train_test.py" -t $test -m $model -lr $lr -bs 1 -d "/data/mjhorry/Experiments/JSRT Nodule Segmentation/input/nodule_files/$test/crossval" -e 25 -a true -x $dim -k 0 -inf True -in True -od "/data/mjhorry/Experiments/JSRT Nodule Segmentation/input/nodule_files/NIH_SEG/crossval" -ext true
 
  python "JSRT_Nodule_Chest_Xray_kfold_train_test.py" -t $test -m $model -lr $lr -bs 1 -d "/data/mjhorry/Experiments/JSRT Nodule Segmentation/input/nodule_files/$test/crossval" -e 25 -a true -x $dim -k 0 -inf True -in True -od "/data/mjhorry/Experiments/JSRT Nodule Segmentation/input/nodule_files/NIH_SEG_A/crossval" -ext true

 done
done

for test in JSRT_SOURCE_HE_SEG_A_EXT JSRT_SOURCE_HE_SEG_B_EXT JSRT_SOURCE_HE_SEG_C_EXT JSRT_SOURCE_HE_SEG_D_EXT
do
 for model in RESUNET-S RESUNET-M RESUNET-L
 do 
# python "JSRT_Nodule_Chest_Xray_kfold_train_test.py" -t $test -m $model -lr $lr -bs 1 -d "/data/mjhorry/Experiments/JSRT Nodule Segmentation/input/nodule_files/$test/crossval" -e 25 -a true -x $dim -k 0 -inf True -in True -od "/data/mjhorry/Experiments/JSRT Nodule Segmentation/input/nodule_files/NIH_HE_SEG/crossval" -ext true
 
 python "JSRT_Nodule_Chest_Xray_kfold_train_test.py" -t $test -m $model -lr $lr -bs 1 -d "/data/mjhorry/Experiments/JSRT Nodule Segmentation/input/nodule_files/$test/crossval" -e 25 -a true -x $dim -k 0 -inf True -in True -od "/data/mjhorry/Experiments/JSRT Nodule Segmentation/input/nodule_files/NIH_HE_SEG_A/crossval" -ext true
 done
done


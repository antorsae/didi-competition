SECONDS=0
T=0
function t()
{
	echo Took $(($SECONDS - $T)) seconds
	T=$SECONDS
}
python refine.py -xi tracklet_labels.xml -xo tracklet_labels_refined.xml -a -f 1/15 ;t
python refine.py -xi tracklet_labels_refined.xml -xo tracklet_labels_trainable.xml -r -f 1/15

python refine.py -xi tracklet_labels.xml -xo tracklet_labels_refined.xml -a -f 1/17 ;t
python refine.py -xi tracklet_labels_refined.xml -xo tracklet_labels_trainable.xml -r -f 1/17

python refine.py -xi tracklet_labels.xml -xo tracklet_labels_refined.xml -a -f 1/21_f ;t
python refine.py -xi tracklet_labels_refined.xml -xo tracklet_labels_trainable.xml -r -f 1/21_f

python refine.py -xi tracklet_labels.xml -xo tracklet_labels_refined.xml -a -f 2/13 ;t
python refine.py -xi tracklet_labels_refined.xml -xo tracklet_labels_trainable.xml -r -f 2/13

python refine.py -xi tracklet_labels.xml -xo tracklet_labels_refined.xml -a -f 2/11_f ;t
# NEED to generate _refined2
python refine.py -xi tracklet_labels_refined2.xml -xo tracklet_labels_trainable.xml -r -f 2/11_f

python refine.py -xi tracklet_labels.xml -xo tracklet_labels_refined.xml -a -f 2/14_f ;t
python refine.py -xi tracklet_labels_refined.xml -xo tracklet_labels_trainable.xml -r -f 2/14_f

python refine.py -xi tracklet_labels.xml -xo tracklet_labels_refined.xml -a -f 2/17 ;t
python refine.py -xi tracklet_labels_refined.xml -xo tracklet_labels_trainable.xml -r -f 2/17

python refine.py -xi tracklet_labels.xml -xo tracklet_labels_refined.xml -a -f 2/3_f ;t
# TODO correct yaw few frames at the beginning
python refine.py -xi tracklet_labels_refined.xml -xo tracklet_labels_trainable.xml -r -f 2/3_f

python refine.py -xi tracklet_labels.xml -xo tracklet_labels_refined.xml -a -f 2/6_f ;t
python refine.py -xi tracklet_labels_refined.xml -xo tracklet_labels_trainable.xml -r -f 2/6_f

python refine.py -xi tracklet_labels.xml -xo tracklet_labels_refined.xml -a -f 2/12_f ;t
# NEED to generate _refined2
python refine.py -xi tracklet_labels_refined2.xml -xo tracklet_labels_trainable.xml -r -f 2/12_f

echo Total time $SECONDS

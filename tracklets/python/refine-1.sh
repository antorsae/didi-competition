SECONDS=0
T=0
function t()
{
	echo Took $(($SECONDS - $T)) seconds
	T=$SECONDS
}
python refine.py -xi tracklet_labels.xml -xo tracklet_labels_refined.xml -a -f 1/8_f ;t
python refine.py -xi tracklet_labels_refined.xml -xo tracklet_labels_trainable.xml -r -f 1/8_f

python refine.py -xi tracklet_labels.xml -xo tracklet_labels_refined.xml -a -f 1/14_f -y 0.89491426 ;t
python refine.py -xi tracklet_labels_refined.xml -xo tracklet_labels_trainable.xml -r -f 1/14_f

python refine.py  -xi tracklet_labels.xml -xo tracklet_labels_refined.xml -a -f 1/13 -y 0.88381232 ;t
python refine.py -xi tracklet_labels_refined.xml -xo tracklet_labels_trainable.xml -r -f 1/13

python refine.py -xi tracklet_labels.xml -xo tracklet_labels_refined.xml -a -f 1/11 -y 0.93886924 ;t
python refine.py -xi tracklet_labels_refined.xml -xo tracklet_labels_trainable.xml -r -f 1/11

python refine.py  -xi tracklet_labels.xml -xo tracklet_labels_refined.xml -a -f 1/10 -y 0.922499 ;t
python refine.py -xi tracklet_labels_refined.xml -xo tracklet_labels_trainable.xml -r -f 1/10

python refine.py  -xi tracklet_labels.xml -xo tracklet_labels_refined.xml -a -f 1/2 -y 0.88522899 ;t
# NEED to generate _refined2 and refined3
python refine.py -xi tracklet_labels_refined3.xml -xo tracklet_labels_trainable.xml -r -f 1/2

python refine.py  -xi tracklet_labels.xml -xo tracklet_labels_refined.xml -a -f 1/3 -y 0.9045834 ;t
python refine.py -xi tracklet_labels_refined.xml -xo tracklet_labels_trainable.xml -r -f 1/3

python refine.py  -xi tracklet_labels.xml -xo tracklet_labels_refined.xml -a -f 1/18 ;t
python refine.py -xi tracklet_labels_refined.xml -xo tracklet_labels_trainable.xml -r -f 1/18

python refine.py  -xi tracklet_labels.xml -xo tracklet_labels_refined.xml -a -f 1/19 ;t
python refine.py -xi tracklet_labels_refined.xml -xo tracklet_labels_trainable.xml -r -f 1/19

python refine.py  -xi tracklet_labels.xml -xo tracklet_labels_refined.xml -a -f 1/20 ;t
python refine.py -xi tracklet_labels_refined.xml -xo tracklet_labels_trainable.xml -r -f 1/20

python refine.py  -xi tracklet_labels.xml -xo tracklet_labels_refined.xml -a -f 1/23 ;t
python refine.py -xi tracklet_labels_refined.xml -xo tracklet_labels_trainable.xml -r -f 1/23

python refine.py  -xi tracklet_labels.xml -xo tracklet_labels_refined.xml -a -f 1/4_f ;t
python refine.py -xi tracklet_labels_refined.xml -xo tracklet_labels_trainable.xml -r -f 1/4_f

python refine.py  -xi tracklet_labels.xml -xo tracklet_labels_refined.xml -a -f 1/6_f ;t
python refine.py -xi tracklet_labels_refined.xml -xo tracklet_labels_trainable.xml -r -f 1/6_f

python refine.py  -xi tracklet_labels.xml -xo tracklet_labels_refined.xml -a -f 1/8_f ;t
python refine.py -xi tracklet_labels_refined.xml -xo tracklet_labels_trainable.xml -r -f 1/8_f

python refine.py  -xi tracklet_labels.xml -xo tracklet_labels_refined.xml -a -f 1/26 ;t
# NEED to generate _refined2
python refine.py -xi tracklet_labels_refined2.xml -xo tracklet_labels_trainable.xml -r -f 1/26

echo Total time $SECONDS


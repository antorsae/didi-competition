SECONDS=0
T=0
function t()
{
	echo Took $(($SECONDS - $T)) seconds
	T=$SECONDS
}
python refine.py -xi tracklet_labels.xml -xo tracklet_labels_refined.xml -a -f 3/2_f ;t
python refine.py -xi tracklet_labels.xml -xo tracklet_labels_refined.xml -a -f 3/12_f ;t
python refine.py -xi tracklet_labels.xml -xo tracklet_labels_refined.xml -a -f 3/13_f ;t
python refine.py -xi tracklet_labels.xml -xo tracklet_labels_refined.xml -a -f 3/14 -ap 0.8;t
python refine.py -xi tracklet_labels.xml -xo tracklet_labels_refined.xml -a -f 3/15_f -ap 0.8;t


# NEED YAW
python refine.py  -xi tracklet_labels.xml -f 3/1  -1 0 -v -a; t
python refine.py  -xi tracklet_labels.xml -f 3/4  -1 0 -v -a; t
python refine.py  -xi tracklet_labels.xml -f 3/6  -1 0 -v -a; t
python refine.py  -xi tracklet_labels.xml -f 3/7  -1 0 -v -a; t
python refine.py  -xi tracklet_labels.xml -f 3/8  -1 0 -v -a; t
python refine.py  -xi tracklet_labels.xml -f 3/9  -1 0 -v -a; t
python refine.py  -xi tracklet_labels.xml -f 3/11_f  -1 0 -v -a; t

echo Total time $SECONDS

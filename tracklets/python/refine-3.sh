SECONDS=0
T=0
function t()
{
	echo Took $(($SECONDS - $T)) seconds
	T=$SECONDS
}
python refine.py -xi tracklet_labels.xml -xo tracklet_labels_refined.xml -a -f 3/2_f ;t


# NEED YAW
python refine.py  -xi tracklet_labels.xml -f 3/1  -1 0 -v -a; t
python refine.py  -xi tracklet_labels.xml -f 3/4  -1 0 -v -a; t
python refine.py  -xi tracklet_labels.xml -f 3/6  -1 0 -v -a; t
python refine.py  -xi tracklet_labels.xml -f 3/7  -1 0 -v -a; t
python refine.py  -xi tracklet_labels.xml -f 3/8  -1 0 -v -a; t
python refine.py  -xi tracklet_labels.xml -f 3/9  -1 0 -v -a; t
python refine.py  -xi tracklet_labels.xml -f 3/11_f  -1 0 -v -a; t

echo Total time $SECONDS

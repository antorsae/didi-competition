SECONDS=0
T=0
function t()
{
	echo Took $(($SECONDS - $T)) seconds
	T=$SECONDS
}

# ok
python refine.py -xi tracklet_labels.xml -xo tracklet_labels_refined.xml -a -f 3/2_f ;t
python refine.py -xi tracklet_labels_refined.xml -xo tracklet_labels_trainable.xml -r -f 3/2_f


# ok needs ransac
python refine.py -xi tracklet_labels.xml -xo tracklet_labels_refined.xml -a -f 3/12_f ;t
python refine.py -xi tracklet_labels_refined.xml -xo tracklet_labels_trainable.xml -r -f 3/12_f

# ok needs ransac
python refine.py -xi tracklet_labels.xml -xo tracklet_labels_refined.xml -a -f 3/13_f ;t
python refine.py -xi tracklet_labels_refined.xml -xo tracklet_labels_trainable.xml -r -f 3/13_f

# ok needs ransac
python refine.py -xi tracklet_labels.xml -xo tracklet_labels_refined.xml -a -f 3/14 -ap 0.85;t
python refine.py -xi tracklet_labels_refined.xml -xo tracklet_labels_trainable.xml -r -f 3/14

# ok needs ransac (bad quality?)
python refine.py -xi tracklet_labels.xml -xo tracklet_labels_refined.xml -a -f 3/15_f -ap 0.85;t
python refine.py -xi tracklet_labels_refined.xml -xo tracklet_labels_trainable.xml -r -f 3/15_f

# ok needs ransac
python refine.py -xi tracklet_labels.xml -xo tracklet_labels_refined.xml -a -f 3/4 -y 0.801401903504; t
python refine.py -xi tracklet_labels_refined.xml -xo tracklet_labels_trainable.xml -r -f 3/4

#ok
python refine.py -xi tracklet_labels.xml -xo tracklet_labels_refined.xml -a -f 3/1 -y 0.95; t
python refine.py -xi tracklet_labels_refined.xml -xo tracklet_labels_trainable.xml -r -f 3/1

# ok
python refine.py -xi tracklet_labels.xml -xo tracklet_labels_refined.xml -a -f 3/6 -y 0.843107080924; t
python refine.py -xi tracklet_labels_refined.xml -xo tracklet_labels_trainable.xml -r -f 3/6

# ok needs ransac
python refine.py -xi tracklet_labels.xml -xo tracklet_labels_refined.xml -a -f 3/7 -y 0.819867282753; t
python refine.py -xi tracklet_labels_refined.xml -xo tracklet_labels_trainable.xml -r -f 3/7

# ok needs ransac
python refine.py -xi tracklet_labels.xml -xo tracklet_labels_refined.xml -a -f 3/8 -y 0.819051722035; t
python refine.py -xi tracklet_labels_refined.xml -xo tracklet_labels_trainable.xml -r -f 3/8

# ok needs ransac
python refine.py -xi tracklet_labels.xml -xo tracklet_labels_refined.xml -a -f 3/9 -y 0.799476375819; t
python refine.py -xi tracklet_labels_refined.xml -xo tracklet_labels_trainable.xml -r -f 3/9

# very few frames are good.
python refine.py -xi tracklet_labels.xml -xo tracklet_labels_refined.xml -a -f 3/11_f -y 0.769471813278; t
python refine.py -xi tracklet_labels_refined.xml -xo tracklet_labels_trainable.xml -r -f 3/11_f

echo Total time $SECONDS

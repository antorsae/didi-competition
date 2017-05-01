mkdir -p /home/antor/cine/didi-ext/release2/Data-points-processed/$1
python bag_to_kitti.py -t obs_rec -L -l -m -i /home/antor/cine/didi-ext/release2/Data-points/$1 -o /home/antor/cine/didi-ext/release2/Data-points-processed/$1 

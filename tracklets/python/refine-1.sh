SECONDS=0
T=0
function t()
{
	echo Took $(($SECONDS - $T)) seconds
	T=$SECONDS
}
python refine.py -i /home/antor/didi-ext/release2/Data-points-processed/ -f 1/8_f ;t
python refine.py -i /home/antor/didi-ext/release2/Data-points-processed/ -f 1/14_f -y 0.89491426 ;t
python refine.py -i /home/antor/didi-ext/release2/Data-points-processed/ -f 1/13 -y 0.88381232 ;t
python refine.py -i /home/antor/didi-ext/release2/Data-points-processed/ -f 1/11 -y 0.93886924 ;t
python refine.py -i /home/antor/didi-ext/release2/Data-points-processed/ -f 1/10 -y 0.922499 ;t
python refine.py -i /home/antor/didi-ext/release2/Data-points-processed/ -f 1/2 -y 0.88522899 ;t
python refine.py -i /home/antor/didi-ext/release2/Data-points-processed/ -f 1/3 -y 0.9045834 ;t
python refine.py -i /home/antor/didi-ext/release2/Data-points-processed/ -f 1/15 ;t
python refine.py -i /home/antor/didi-ext/release2/Data-points-processed/ -f 1/17 ;t
python refine.py -i /home/antor/didi-ext/release2/Data-points-processed/ -f 1/18 ;t
python refine.py -i /home/antor/didi-ext/release2/Data-points-processed/ -f 1/19 ;t
python refine.py -i /home/antor/didi-ext/release2/Data-points-processed/ -f 1/20 ;t
python refine.py -i /home/antor/didi-ext/release2/Data-points-processed/ -f 1/23 ;t
python refine.py -i /home/antor/didi-ext/release2/Data-points-processed/ -f 1/4 ;t
python refine.py -i /home/antor/didi-ext/release2/Data-points-processed/ -f 1/6 ;t
python refine.py -i /home/antor/didi-ext/release2/Data-points-processed/ -f 1/8 ;t
python refine.py -i /home/antor/didi-ext/release2/Data-points-processed/ -f 1/26 ;t
echo Total time $SECONDS


SECONDS=0
T=0
function t()
{
	echo Took $(($SECONDS - $T)) seconds
	T=$SECONDS
}
python refine.py -i /home/antor/didi-ext/release2/Data-points-processed/ -f 1/15 ;t
python refine.py -i /home/antor/didi-ext/release2/Data-points-processed/ -f 1/17 ;t
python refine.py -i /home/antor/didi-ext/release2/Data-points-processed/ -f 1/21_f ;t
python refine.py -i /home/antor/didi-ext/release2/Data-points-processed/ -f 2/13 ;t
python refine.py -i /home/antor/didi-ext/release2/Data-points-processed/ -f 2/11_f ;t
python refine.py -i /home/antor/didi-ext/release2/Data-points-processed/ -f 2/14_f ;t
python refine.py -i /home/antor/didi-ext/release2/Data-points-processed/ -f 2/17 ;t
python refine.py -i /home/antor/didi-ext/release2/Data-points-processed/ -f 2/3_f ;t
python refine.py -i /home/antor/didi-ext/release2/Data-points-processed/ -f 2/6_f ;t
python refine.py -i /home/antor/didi-ext/release2/Data-points-processed/ -f 2/12_f ;t

echo Total time $SECONDS

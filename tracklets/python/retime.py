import rosbag
import rospy
import argparse

parser = argparse.ArgumentParser(description='Convert rosbag to images and csv.')
parser.add_argument('-i', '--ibag', type=str, nargs='?', required=True, help='input bag to process')
parser.add_argument('-o', '--obag', type=str, default ='/dev/null', nargs='?', help='output bag')
parser.add_argument('-t', '--topics', type=str, nargs='+', default=None, help='topics to filter')
parser.add_argument('-s', '--seconds', type=float, default =0.1, nargs='?', help='time threshold in seconds, default: 0.1')
args = parser.parse_args()
filter_topics = args.topics
seconds       = args.seconds
obag          = args.obag
ibag          = args.ibag

duration = rospy.rostime.Duration(secs = int(seconds//1), nsecs = int((seconds % 1) * 1.e9))

with rosbag.Bag(obag, 'w') as outbag:
    for topic, msg, t in rosbag.Bag(ibag).read_messages():
        newstamp = msg.header.stamp if msg._has_header else t
        if filter_topics is None or topic in filter_topics:
                if msg._has_header:
                    diff = msg.header.stamp - t
                    if abs(diff) >= duration:
                        print(topic + " @" +  str(msg.header.seq) + " is " +  str(abs(diff.to_sec())) + " seconds off")
                        newstamp = t
                    msg.header.stamp = newstamp
        if obag is not '/dev/null':
            outbag.write(topic, msg, t)

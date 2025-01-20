#!/bin/bash

if [ -z "$1" ]
  then
    echo "Usage: $0 settings_file"
    exit 1
fi

RUN_PATH="/storage01/grexai/dev/mrcnngit/FinalModel/g_run.sh"

# new
maxVal1=51
startVal1=21
maxVal2=76
startVal2=41
DETECTION_NMS_THRESHOLD=$startVal1
RPN_NMS_THRESHOLD=$startVal2

for ((i=$startVal1;i<=$maxVal1;i=i+1))
do
	DETECTION_NMS_THRESHOLD=$i
	echo "DETECTION_NMS_THRESHOLD: $i"
	source $RUN_PATH $1 $DETECTION_NMS_THRESHOLD $RPN_NMS_THRESHOLD
	for ((j=$startVal2;j<=$maxVal2;j=j+1))
	do
		RPN_NMS_THRESHOLD=$j
		echo "RPN_NMS_THRESHOLD: $j"
		source $RUN_PATH $1 $DETECTION_NMS_THRESHOLD $RPN_NMS_THRESHOLD
	done
done



# Non-maxValimum suppression threshold for detection
#DETECTION_NMS_THRESHOLD = 0.3

# Non-maxVal suppression threshold to filter RPN proposals.
# You can reduce this during training to generate more propsals.
#RPN_NMS_THRESHOLD = 0.7

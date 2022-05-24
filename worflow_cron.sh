#!/bin/sh

# Change the output directory here!! 
MONITORDIR="/xyx/amarolab_dcd_data_dir/"

echo "LOG FILE: Tracking files created in this directory" > ./logs/directory_tracker_spike.log

<path>/inotifywait -mr -e create --format '%w%f' "${MONITORDIR}" | while read NEWFILE
do	
	echo ${NEWFILE#${MONITORDIR}/} >> ./logs/directory_tracker_spike.log
	ts=`date +%Y-%m-%d:%H:%M:%S`
	PATHTONEWFILE=`echo ${NEWFILE#${MONITORDIR}/}` 
	#Check if new version is created, format for directory --> dcdDataSet_ID/open/file
    
	ISNEWVERSION=`echo $PATHTONEWFILE | cut -f3 -d "/"`
	echo $ts $PATHTONEWFILE >> ./logs/directory_tracker_spike.log
	echo $ts $ISNEWVERSION >> ./logs/directory_tracker_spike.log
	if [ -n "$ISNEWVERSION" ]; then
		echo "Calling python script $PATHTONEWFILE $NEWFILE" 
#		<path>/python dcd_featureExtract.py $NEWFILE> ./logs/featureExtract_$ts.log
	fi
done
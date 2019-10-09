#!/usr/bin/env bash
# Executing cloudcompare.CloudCompare from remote returns following error, although X11 forwarding is working well with other soft (xclock, pycharm, ...):
# QXcbConnection: Could not connect to display localhost:11.0
# Abandon (core dumped)
# Could not find any solution, thus this script must be executed on the local plateform.
# 

#Number of division on the circle
nbDiv=4
#Height of the scanner (cm)
hScan=1.5
#Number of model to process
numberOfTrees=10
for n in $( seq 1 $numberOfTrees)
do
    #Radius of the circle (m)
    for r in 10
    do
        for div in $( seq 1 $nbDiv )
        do
            #Compute the coordinates of the camera
            fullAng=$(echo "scale=10; 8*a(1)" | bc -l)
            angFrac=$(echo "scale=10; $fullAng / $nbDiv" | bc -l)
            ang=$(echo "scale=10; $div*$angFrac" | bc -l)
            OX=$(echo "scale=3; $r*c($ang)" | bc -l)
            OY=$(echo "scale=3; $r*s($ang)" | bc -l)
            OZ=$hScan
            fullNoExt="full${n}_${OX}_${OY}_${OZ}"
            fullIn="${fullNoExt}.xyz"
            fullOut="${fullNoExt}_subsampled.xyz"


            cloudcompare.CloudCompare -SILENT -O $fullIn -C_EXPORT_FMT ASC -SS SPATIAL 0.005

            patOut="${fullNoExt}*.asc"
            find . -type f -name "${fullNoExt}*.asc" -exec mv {} "${fullOut}" \;
        done
    done
done




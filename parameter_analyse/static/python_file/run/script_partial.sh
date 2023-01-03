#! /bin/bash

cd $1
ls
name_file_spike=$4"_spike_recorder"
echo $name_file_spike
# not overwrite a previous file
if [ -f $name_file_spike'_ex.dat' ]
then
    exit 1
fi
echo "save file :" $name_file_spike'_ex.dat'
#head -n 3 spike_recorder-$2-00.dat >> $name_file_spike'_ex.dat'
for i in spike_recorder-$2-*.dat
do
  echo $i
  tail -n $(expr $(wc -l $i |cut -f 1 -d\ ) - 3) $i >> $name_file_spike'_ex.dat'
	rm $i
done
#head -n 3 spike_recorder-$3-00.dat >> $name_file_spike'_in.dat'
for i in spike_recorder-$3-*.dat
do
  echo $i
  tail -n $(expr $(wc -l $i |cut -f 1 -d\ ) - 3) $i >> $name_file_spike'_in.dat'
	rm $i
done
cd ..

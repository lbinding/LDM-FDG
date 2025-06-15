#!/bin/bash -l

# Batch script 

# Request Time (format hours:minutes:seconds).
#$ -l h_rt=999:0:0
# Request RAM per core. 
#$ -l gpu=true,gpu_type=rtx4090
#$ -l tmem=24G

#$ -o /home/lbinding/p002_deepSuStaIn/ # folder to store the shell outputs

#Set Name 
#$ -N ldm_fdg
date
hostname

#Setup python 
source /share/apps/source_files/python/python-3.9.5.source

# Loop over arguments looking for -i and -o
args=("$@")
i=0
while [ $i -lt $# ]; do
    #Set npy data 
    if ( [ ${args[i]} = "-script" ] ) ; then
      let i=$i+1
      script=${args[i]}
    fi
    let i=$i+1
done

# Check if user gave correct inputs
if [[ -z "${script}" ]]; then
    correct_input=0
else 
    correct_input=1
fi

#Check the user has provided the correct inputs
if ( [[ ${correct_input} -eq 0 ]] ) ; then
  echo ""
  echo "Incorrect input. Please see below for correct use"
  echo ""
  echo "Options:"
  echo " -script:           Input script -- REQUIRED"
  echo ""
  echo "${script_name} -script python.py"
  echo ""
  exit
fi

echo "Calling Python Script"
python3 ${script}
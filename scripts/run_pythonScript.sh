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

#Default vars:
train_vAE = ""
train_LDM = ""



# Loop over arguments looking for -i and -o
args=("$@")
i=0
while [ $i -lt $# ]; do
    #Set npy data 
    if ( [ ${args[i]} = "-script" ] ) ; then
      let i=$i+1
      script=${args[i]}
    elif ( [ ${args[i]} = "-train_vAE" ] ) ; then
      train_vAE="--train_vAE"
    elif ( [ ${args[i]} = "-train_LDM" ] ) ; then
      train_LDM="--train_LDM"
    elif ( [ ${args[i]} = "-vAE_model" ] ) ; then
      let i=$i+1
      vAE_model=${args[i]}
    elif ( [ ${args[i]} = "-LDM_model" ] ) ; then
      let i=$i+1
      LDM_model=${args[i]}
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
  echo " -script:           PATH to Input script -- REQUIRED"
  echo " -train_vAE:        FLAG (sets training to True) -- OPTIONAL"
  echo " -train_LDM:        FLAG (sets training to True) -- OPTIONAL"
  echo " -vAE_model:        PATH to weights file for vAE -- OPTIONAL"
  echo " -LDM_model:        PATH to weights file for LDM -- OPTIONAL"
  echo ""
  echo "${script_name} -script python.py"
  echo ""
  exit
fi

# Add --vAE_model at the start of the variable vAE_model if it is set
if [[ -n "${vAE_model}" ]]; then
  vAE_model="--vAE_model ${vAE_model}"
  else
  vAE_model = ""
fi

# Add --LDM_model at the start of the variable LDM_model if it is set
if [[ -n "${lDM_model}" ]]; then
  LDM_model="--LDM_model ${LDM_model}"
  else
  LDM_model = ""
fi

echo "Calling Python Script"
python3 "${script}" ${train_vAE} ${train_LDM} "${vAE_model}" "${LDM_model}"

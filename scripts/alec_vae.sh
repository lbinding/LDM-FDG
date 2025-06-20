#==================================================
# Basic configurations
#==================================================

#$ -l tmem=23.8G
#$ -l h_rt=120:00:00

#==================================================
# Optional configurations
#==================================================

#$ -S /bin/bash
#$ -j y
#$ -N demo_ldm
#$ -wd /cluster/project2/CU-MONDAI/Alec_Tract/LDM-FDG
#$ -o  /cluster/project2/CU-MONDAI/Alec_Tract/logs
#$ -e  /cluster/project2/CU-MONDAI/Alec_Tract/logs

#==================================================
# Special configurations
#==================================================

#$ -l gpu=true,gpu_type=(a100_80|h100)

#========================================================================================
# Run the script
#========================================================================================

source /share/apps/source_files/python/python-3.11.9.source
source /cluster/project2/CU-MONDAI/Alec_Tract/mri_pet/new_venv/bin/activate

nvidia-smi



#Default vars
script_name="/cluster/project2/CU-MONDAI/Alec_Tract/LDM-FDG/LDM-FDG/Main.py"
train_vAE=""
train_LDM=""
help_true=0

# Loop over arguments looking for -i and -o
args=("$@")
i=0
while [ $i -lt $# ]; do
    #Set npy data 
    if ( [ ${args[i]} = "-help" ] ) ; then
      help_true=1
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

# Print help for running the script 
if [[ ${help_true} -eq 1 ]]; then
  echo ""
  echo "Incorrect input. Please see below for correct use"
  echo ""
  echo "Options:"
  echo " -help:             Display this help message."
  echo " -train_vAE:        FLAG (sets training to True) -- OPTIONAL"
  echo " -train_LDM:        FLAG (sets training to True) -- OPTIONAL"
  echo " -vAE_model:        PATH to weights file for vAE -- OPTIONAL"
  echo " -LDM_model:        PATH to weights file for LDM -- OPTIONAL"
  echo ""
  echo "${script_name} -train_vAE -train_LDM -vAE_model /Path/To/Weights/File.pt -LDM_model /Path/To/Weights/File.pt"
  echo ""
  exit 0 
fi

# Add --vAE_model at the start of the variable vAE_model if it is set
if [[ -n "${vAE_model}" ]]; then
    vAE_MODEL_ARGS=("--vAE_model" "${vAE_model}")
else
    vAE_MODEL_ARGS=()
fi
# Add --LDM_model at the start of the variable LDM_model if it is set
if [[ -n "${LDM_model}" ]]; then
    LDM_MODEL_ARGS=("--LDM_model" "${LDM_model}")
else
    LDM_MODEL_ARGS=()
fi

# Use an array to collect all arguments for safety.
PYTHON_ARGS=()
[[ -n "${train_vAE}" ]] && PYTHON_ARGS+=("${train_vAE}")
[[ -n "${train_LDM}" ]] && PYTHON_ARGS+=("${train_LDM}")
PYTHON_ARGS+=("${vAE_MODEL_ARGS[@]}")
PYTHON_ARGS+=("${LDM_MODEL_ARGS[@]}")


echo "Calling Python Script"
echo "${script_name}  ${PYTHON_ARGS[@]}"
python3 "${script_name}" "${PYTHON_ARGS[@]}"


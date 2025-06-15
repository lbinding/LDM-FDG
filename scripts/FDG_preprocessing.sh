#FDG preprocessing 


for subID in *; do
    echo ${subID}
    cd ${subID}
    # Step 1: Get the DICOM folder containing the first .dcm
    dcm_dir=$(dirname "$(find . -maxdepth 4 -type f -name "*.dcm" -print -quit)")

    # Step 2: Get parent folder of dcm_dir (where output should be stored)
    out_dir=$(dirname "$dcm_dir")

    # Step 3: Get the top-level folder (to clean commas from)
    top_level=$(echo "$out_dir" | cut -d/ -f2)  # e.g., "Coreg,_Avg,_Std_Img_and_Vox_Siz,_Uniform_Resolution"
    clean_name=$(echo "$top_level" | tr -d ',' | tr ' ' '_')  # Remove commas and replace spaces

    # Step 4: Compose full output path & convert
    dcm2niix -z y -f "${clean_name}_$(basename "$dcm_dir")" -o "$out_dir" "$dcm_dir"

    # Step 5: Clean filename 
    mv ${top_level} ${clean_name}

    # Step 6: Rename the output directory
    parent_dir=$(dirname "$out_dir")
    basename_old=$(basename "$out_dir")

    # Convert to 2011_06_09_082348 using sed
    basename_new=$(echo "$basename_old" | sed -E 's/-/_/g; s/_([0-9]{2})_([0-9]{2})_([0-9]{2})\..*/_\1\2\3/')

    # Rename the directory
    mv ${clean_name}/${basename_old} ${clean_name}/${basename_new}
    cd ../
done 

for subID in *; do
    if [ -d ${subID} ]; then 
        echo ${subID}
        cd ${subID}
        nii_file=`find . -maxdepth 4 -type f -name "*.nii.gz" -print -quit`
        echo `mrinfo ${nii_file} | grep "Dimensions:"`
        mrgrid ${nii_file} regrid -size 240,240,96 ${nii_file//.nii.gz}_regrid.nii.gz 
        cd ../
    fi 
done 

for subID in *; do
    if [ -d ${subID} ]; then 
        echo ${subID}
        cd ${subID}
        nii_file=`find . -maxdepth 4 -type f -name "*_regrid.nii.gz" -print -quit`
        fslroi ${nii_file} ${nii_file//.nii.gz}_slice25.nii.gz 0 -1 0 -1 25 1
        fslroi ${nii_file} ${nii_file//.nii.gz}_slice40.nii.gz 0 -1 0 -1 40 1
        fslroi ${nii_file} ${nii_file//.nii.gz}_slice65.nii.gz 0 -1 0 -1 65 1
        cd ../
    fi
done 

github_repo="/Users/lawrencebinding/Desktop/projects/github/LDM-FDG/data/slices"
for subID in *; do
    if [ -d ${subID} ]; then
        nii_file_slice25=`find . -maxdepth 4 -type f -name "*_regrid_slice25.nii.gz" -print -quit`
        nii_file_slice40=`find . -maxdepth 4 -type f -name "*_regrid_slice40.nii.gz" -print -quit`
        nii_file_slice65=`find . -maxdepth 4 -type f -name "*_regrid_slice65.nii.gz" -print -quit`
        cp -a ${nii_file_slice25} ${github_repo}/${subID}_slice25.nii.gz
        cp -a ${nii_file_slice40} ${github_repo}/${subID}_slice40.nii.gz
        cp -a ${nii_file_slice65} ${github_repo}/${subID}_slice65.nii.gz
    fi
done 

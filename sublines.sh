# submit jobs

# waterfall for 3 datasets
for num in {1..20}
do
    for gp in "paincontrol" "paintype" "digestive"
    do 
    echo "submitted waterfall task $gp, number $num"
    fsl_sub -T 200 python waterfall_idp.py $gp $num
    done
done
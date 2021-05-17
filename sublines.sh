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

# connectivity mat 
for clf in "rforest" "lgb"
do 
    for conn in "fullcorr_100" "fullcorr_25" "parcorr_100" "parcorr_25" "compamp_100" "compamp_25"
    do
    echo "submitted connmat with clf $clf, connectivity $conn"
    fsl_sub -T 30 python connectivity_mat.py $clf $conn
    done
done
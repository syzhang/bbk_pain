# trying out retrive some basics 
datadir=/vols/Data/pain/asdahl/uk_biobank/Oct2020

# getting gender and age
fsl_sub -T 30 funpack \
    -s ../funpack_cfg/subjs_with_condition.txt \
    -v 31 \
    $datadir/sandbox_suyi/age_gender_with_condition.tsv \
    $datadir/ukb44014.csv $datadir/ukb44219.csv
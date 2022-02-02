# Directory which contains ukb44014.csv, ukb44181.csv (not used), and ukb44219.csv
# datadir=/vols/Scratch/ukbiobank/paulmc/anushka_andy
# datadir=/vols/Data/pain/asdahl/uk_biobank/Nov2020/from_paul
datadir=/vols/Data/pain/asdahl/uk_biobank/jan2022
# outputdir=/vols/Data/pain/asdahl/uk_biobank/suyi_extend/rerun
# outputdir=/vols/Data/pain/asdahl/uk_biobank/suyi_extend/fibro
outputdir=/vols/Data/pain/asdahl/uk_biobank/suyi_extend/pain_questionnaire

# It turns out that FUNPACK has a limitation whereby subject conditional
# expressions cannot be evaluated if the variables used in the expression
# are contained in different files. I will look into this and see if I can
# remove this limitation.
#
# In the meantime, we can work around this by using FUNPACK to create a
# temporary file which contains all of the variables we wish to use to
# select our subjects.

# funpack -nj 4 -n -n -n -wl \
#   -v 25010,20002,6159,3799,4067,3404,3571,3741,3414,3773,2956 \
#   $outputdir/vars_for_query.tsv \
#   $datadir/ukb44014.csv $datadir/ukb44219.csv

# Now we can query the above file to identify subjects in our two groups.
# Storing the query strings in variables for readability.
#
# First, we want a list of subjects who:
#   - Have brain MRI imaging data (25010 != na), and
#   - Meet any of the following criteria:
#     - 20002 == 1154
#     - 20002 == 1265
#     - 20002 == 1313
#     - 20002 == 1477
#     - 20002 == 1464
#     - 20002 == 1465
#     - 20002 == 1542
#     - 2956  == 1 (general pain 3+m)
#     (extended by adding other pain conditions)
#     - 3799  == 1 (headache 3+m)
#     - 4067  == 1 (facial pain 3+m)
#     - 3404  == 1 (neck/shoulder pain 3+m)
#     - 3571  == 1 (back pain 3+m)
#     - 3741  == 1 (stomach pain 3+m)
#     - 3414  == 1 (hip pain 3+m)
#     - 3773  == 1 (knee pain 3+m)

# subjs_with_condition="v25010 != na && (v20002 == 1154 || v20002 == 1265 || v20002 == 1313 || v20002 == 1477 || v20002 == 1464 || v20002 == 1465 || v20002 == 1542 || v2956 == 1 || v3799 == 1 || v4067 == 1 || v3404 == 1 || v3571 == 1 || v3741 == 1 || v3414 == 1 || v3773 == 1)"

# Run the above queries
# funpack -n -n -n -wl -s "$subjs_with_condition"    $outputdir/subjs_with_condition.tsv    $outputdir/vars_for_query.tsv

# The sole purpose of the above FUNPACK calls is to identify subjects that
# meet our criteria. So now we extract the subject IDs from each of
# the queries and store them in plain text files, with one subject ID
# on each line. We use "cut" to extract the first column, and "tail" to
# drop the first row (containing column names):
# cat $outputdir/subjs_with_condition.tsv    | cut -f 1 | tail -n+2 > $outputdir/subjs_with_condition.txt

# Now we can use these files in future FUNPACK runs to query the
# original data. For example, if we are interested in variable
# 12345 in subjects with any of the conditions:
# (using subjects extracted with pandas)

# # extract clinical and idp vars from all visits
# funpack \
#     -s $outputdir/subjs_disease_allvisits_extended.csv \
#     -v $outputdir/clinical_idp_variables.txt \
#     $outputdir/qsidp_subjs_disease_allvisits_extended.tsv \
#     $datadir/ukb44014.csv $datadir/ukb44219.csv

# # extract clinical and idp vars from visit2 pain subjects
# funpack \
#     -s $outputdir/subjs_disease_visit2_extended.csv \
#     -v $outputdir/clinical_idp_variables.txt \
#     $outputdir/qsidp_subjs_disease_visit2_extended.tsv \
#     $datadir/ukb44014.csv $datadir/ukb44219.csv

# ## extract clinical and idp vars from all visits control subjects
# funpack \
#     -s $outputdir/subjs_without_condition_cwpfree.txt \
#     -v $outputdir/clinical_idp_variables.txt \
#     $outputdir/qsidp_subjs_control_visit2_extended.tsv \
#     $datadir/ukb44014.csv $datadir/ukb44219.csv

# # extract clinical and idp vars from matched control subjects
# funpack \
#     -s $outputdir/subjs_controls_visit2_matched.csv \
#     -v $outputdir/clinical_idp_variables.txt \
#     $outputdir/qsidp_subjs_control_visit2_matched.tsv \
#     $datadir/ukb44014.csv $datadir/ukb44219.csv

# # extract clinical and idp vars from digestive after imaging group
# funpack \
#     -s $outputdir/subjs_digestive_after_imaging.csv \
#     -v $outputdir/clinical_idp_variables.txt \
#     $outputdir/qsidp_subjs_digestive_imaging.tsv \
#     $datadir/ukb44014.csv $datadir/ukb44219.csv


################################################
### using rerun subjects
################################################
# # extract clinical and idp vars from all visits
# funpack \
#     -s $outputdir/subjs_with_condition.txt \
#     -v $outputdir/clinical_idp_variables.txt \
#     $outputdir/qsidp_subjs_disease_allvisits_extended.tsv \
#     $datadir/ukb44014.csv $datadir/ukb44219.csv

# ## extract clinical and idp vars from all visits control subjects
# funpack \
#     -s $outputdir/subjs_without_condition.txt \
#     -v $outputdir/clinical_idp_variables.txt \
#     $outputdir/qsidp_subjs_control_allvisits_extended.tsv \
#     $datadir/ukb44014.csv $datadir/ukb44219.csv

###############################################
### fibro and controls
################################################# 
# extract clinical and idp vars from all visits control subjects
    # -v 20003,137,6154,10004,21000 \
# funpack \
#     -s $outputdir/fibro_subjects.txt \
#     -v 20544,6142 \
#     $outputdir/extra_fields_2_fibro_subjects.tsv \
#     $datadir/ukb44014.csv $datadir/ukb44219.csv

####additional data request on 04/10/21
# funpack \
#     -s $outputdir/participant_eids_041021.txt \
#     -v 25001,25002,25005,25006,25009,25010 \
#     $outputdir/extra_fields_fibro_participant_eids_041021.tsv \
#     $datadir/ukb44014.csv $datadir/ukb44219.csv


###############################################
### pain questionnaire
###############################################

####extract 129 pain questionnaire fields
funpack \
    -s $outputdir/subjs_without_condition.txt \
    -v $datadir/pain_questions_code.csv \
    $outputdir/pain_questionnaire_without_condition.tsv \
    $datadir/ukb49933.csv
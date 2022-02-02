# directory
datadir=/vols/Data/pain/asdahl/uk_biobank/Nov2020/from_paul
outputdir=/vols/Data/pain/asdahl/uk_biobank/suyi_extend/fibro

####additional data request on 04/10/21
funpack \
    -s $outputdir/participant_eids_041021.txt \
    -v 25001,25002,25005,25006,25009,25010 \
    $outputdir/extra_fields_fibro_participant_eids_041021.tsv \
    $datadir/ukb44014.csv $datadir/ukb44219.csv
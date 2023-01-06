#!/usr/bin/env bash

set -o errexit
set -o nounset
set -o pipefail


# processed=./weed_test_set_blob_path.txt
# download=SAS_download_key.txt

# while read line; do
# while read key; do
# src="https://weedsimagerepo.blob.core.windows.net/semifield-developed-images/NC_2022-10-17/images/NC_1666013840.jpg?sv=2020-08-04&st=2022-12-13T23%3A30%3A31Z&se=2022-12-14T23%3A30%3A31Z&sr=c&sp=rl&sig=54khiwVR1%2BPcKnnfa%2Bllg9NO%2B3zTwyg1Rd%2B8fCQNZe8%3D"
dest="https://weedsimagerepo.blob.core.windows.net/semifield-utils?sv=2020-08-04&st=2022-12-14T00%3A04%3A13Z&se=2022-12-15T00%3A04%3A13Z&sr=c&sp=rcwl&sig=euS8emQA3MkAe5ucEfa5WYfmTOgUssIlPNR%2F7EX6NsE%3D"
# echo $src
# prefix=${key%%semifield-developed-images*}
# suffix=${key#*semifield-developed-images} 
# src=${prefix-}semifield-developed-images/${line-}${suffix-}
# src=${prefix-}semifield-developed-images/NC_2022-10-17/images/NC_1666013840.jpg 

# dest=./weed_test_set/${line-}
# dest=./weed_test_set/NC_2022-10-17/images/NC_1666013840.jpg
src=./weed_test_set.csv
# DIR=${dest%/*}
# if [ ! -d $DIR ]; then
#     mkdir -p $DIR
# fi
azcopy copy ${src} ${dest}

# done <$download

# done <$processed

#!/bin/sh
set -e

if [ $# -ne 1 ]; then
    echo "\033[31mERROR\033[0m"\
         "wrong number of arguments"
    echo "\tusage: check_format.sh <DIR>\n"
    exit 1
fi

# Setup some variables
USER_ID=`id -u`
GROUP_ID=`id -g`

docker run --rm -ti \
       -v $PWD:/work_dir:rw \
       --user "${USER_ID}:${GROUP_ID}" \
       -w "/work_dir" \
       ghcr.io/acts-project/format10:v41 \
       .github/check_format.sh $1

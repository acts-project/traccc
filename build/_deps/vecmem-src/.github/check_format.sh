#!/bin/sh
#
# check that all code complies w/ the clang-format specification
#
# if all is well, returns w/o errors and does not print anything.
# otherwise, return an error and print offending changes

set -e # abort on error


if [ $# -ne 1 ]; then
    echo "wrong number of arguments"
    echo ""
    echo "usage: check_format <DIR>"
    exit 1
fi

_binary=${CLANG_FORMAT_BINARY:-clang-format}

$_binary --version

cd $1
find . \( -iname '*.cpp' -or -iname '*.hpp' -or -iname '*.ipp' -or -iname '*.cu' -or -iname '*.cuh' -or -iname '*.hip' -or -iname '*.sycl' \) \
       -and -not -path "./*build*/*" \
       -and -not -path "./thirdparty/*" \
  | xargs $_binary -i -style=file


if ! [ -z $CI ] || ! [ -z $GITHUB_ACTIONS ]; then
  mkdir changed
  for f in $(git diff --name-only); do
    cp --parents $f changed
  done
fi
    
echo "clang-format done"
    
set +e
git diff --exit-code --stat
result=$?

if [ "$result" -eq "128" ]; then
    echo "Format was successfully applied"
    echo "Could not create summary of affected files"
    echo "Are you in a submodule?"

fi

exit $result

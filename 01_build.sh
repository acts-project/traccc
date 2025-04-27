rm -rf $HOME/project/traccc/build
mkdir -p $HOME/project/traccc/build 
cmake -S $HOME/project/traccc -B $HOME/project/traccc/build \
    --preset cuda-fp32 \
    -DTRACCC_BUILD_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES="75" \
    -DTRACCC_BUILD_TESTING=ON \
    -DTRACCC_BUILD_EXAMPLES=ON \
    -DCMAKE_CXX_STANDARD=20 \
    -DTRACCC_USE_ROOT=ON | tee build01.log

cmake --build $HOME/project/traccc/build -- -j16 2>&1 | tee build02.log

#grep -P "ptxas info.*Used" build02.log | \
#  sed -e 's/^.*ptxas info/ptxas info/' | sort -u | tee ptxas.log
#
## 列出所有有 spill 的行
#grep -E "bytes spill (stores|loads)" build02.log | tee spill_summary.log
#
## 若要看完整上下文（包含該 kernel 名稱、register count）
#grep -nE "ptxas info.*Function properties" -n build02.log \
#  | cut -d: -f1 | while read L; do
#      sed -n "$((L-2)),$((L+3))p" build02.log \
#        | grep -E "spill|Used [0-9]+ registers" && echo
#  done | tee spill.log

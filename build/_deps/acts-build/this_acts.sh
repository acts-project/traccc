# set up environment variables
# the ${VAR:+:} part adds a double colon only if VAR is not empty
export PATH="/usr/local/bin${PATH:+:}${PATH}"
export LD_LIBRARY_PATH="/usr/local/lib${LD_LIBRARY_PATH:+:}${LD_LIBRARY_PATH}"
export DYLD_LIBRARY_PATH="/usr/local/lib${DYLD_LIBRARY_PATH:+:}${DYLD_LIBRARY_PATH}"

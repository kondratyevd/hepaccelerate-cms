voms-proxy-init --voms cms
export VOMS_PATH=$(echo $(voms-proxy-info | grep path) | sed 's/path.*: //')
export VOMS_TRG=./$(echo $(voms-proxy-info | grep path) | sed 's/.*tmp\///')
cp $VOMS_PATH $VOMS_TRG
export X509_USER_PROXY=$VOMS_TRG

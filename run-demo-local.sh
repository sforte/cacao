./local-helper.sh distopt.driver \
--trainFile=/Users/simone/X_mean_10_mins.svm \
--testFile=data/regr_test.dat \
--numFeatures=4000 \
--numRounds=50 \
--localIterFrac=1.0 \
--numSplits=4 \
--lambda=0.01 \
--justCoCoA=true \
--beta=1.0
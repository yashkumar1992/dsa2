


RMDIR /S /Q  data/output/titanic/ >nul 2>&1

MKDIR  data\output\titanic  >nul 2>&1


del zlog/log_titanic_prepro.txt >nul 2>&1

del  zlog/log_titanic_train.txt  >nul 2>&1

del  zlog/log_titanic_predict.txt >nul 2>&1


  python titanic_classifier.py  preprocess    > zlog/log_titanic_prepro.txt 2>&1
  python titanic_classifier.py  train    > zlog/log_titanic_train.txt 2>&1
  python titanic_classifier.py  predict  > zlog/log_titanic_predict.txt 2>&1



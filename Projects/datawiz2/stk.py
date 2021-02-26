# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# load data
train = pd.read_csv('clean/X_train.csv')
targets = pd.read_csv('clean/y_train.csv')
test = pd.read_csv('clean/X_test.csv')

# Set up level 1 models...
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB 
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier


RANDOM_SEED = 16

lgbm = LGBMClassifier( #objective='binary', # see this
                      boosting_type='gbdt',
                      n_estimators=1000,
                      metric='auc',
                      learning_rate=0.009,
                      num_leaves=8,
                      feature_fraction=0.5,
                      bagging_fraction=0.5,
                      bagging_freq=1,
                      max_depth=3,
                      reg_alpha=0.75,
                      reg_lambda=0.75)
rf = RandomForestClassifier(n_estimators=500, random_state=RANDOM_SEED)
nb = GaussianNB()
svc = SVC(kernel='rbf', C=1.0, gamma='auto', probability=True)
knn = KNeighborsClassifier(n_neighbors=63)
sgd = SGDClassifier(eta0=1, max_iter=1000, tol=0.0001, alpha=0.01, l1_ratio=1.0, learning_rate='adaptive', loss='log', penalty='elasticnet')


# set up the meta classifier (level 2 model)
from sklearn.linear_model import LogisticRegression
from mlxtend.classifier import StackingCVClassifier
np.random.seed(RANDOM_SEED)
lr = LogisticRegression(max_iter=1000, class_weight='balanced', penalty='l1', C=0.1, solver='liblinear')
sclf = StackingCVClassifier(classifiers=[knn, rf, nb, svc, sgd, lgbm], 
                            use_probas=True,
                            use_features_in_secondary=True,
                            meta_classifier=lr,
                            cv=6)

sclf.fit(train, targets)
preds = sclf.predict(test)
print(preds)
exit(0)


# Set up K-Fold cross validation and predictions
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

num_folds = 6
folds = KFold(n_splits=num_folds, shuffle=True)

test_result = np.zeros(len(test))
auc_score = 0

for fold_, (trn_idx, val_idx) in enumerate(folds.split(train, targets)):
    print("Fold: ", fold_ + 1)
    
    X_train, y_train = train.iloc[trn_idx], targets.iloc[trn_idx]
    X_valid, y_valid = train.iloc[val_idx], targets.iloc[val_idx]
    
    sclf.fit(X_train.values, y_train.values)
    
    y_pred = sclf.predict_proba(X_valid)
    auc = roc_auc_score(y_valid, y_pred[:, 1])
    print(auc)
    auc_score += auc

    preds = sclf.predict_proba(test)
    test_result += preds[:, 1]

# print the average AUC across the folds and compute the final results on the test data
auc_score = auc_score / folds.n_splits
print("AUC score: ", auc_score)
test_result = test_result / folds.n_splits

# create the submission
submission = pd.DataFrame({
  'Id' : test['Id'],
  'Occupation' : test_result
  })
submission.to_csv('simple_stacking.csv', index=False)
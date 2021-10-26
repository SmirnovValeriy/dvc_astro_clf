import sys
import joblib
import json
import lightgbm as lgb
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score

if len(sys.argv) != 5:
    sys.stderr.write("Arguments error. Usage:\n")
    sys.stderr.write("\tpython3 validate.py"
                     "model-pkl-path robust-pkl-path"
                     "df_test-pkl-path scores-json-path\n")
    sys.exit(1)
gb = joblib.load(sys.argv[1])
robust = joblib.load(sys.argv[2])
test_d = joblib.load(sys.argv[3])
scores_file = sys.argv[4]

X, y = test_d[:, :-1], test_d[:, -1].astype('int')
X_norm = robust.transform(X)
y_pred = gb.predict(X_norm)
gb_test_acc = accuracy_score(y, y_pred)
with open(scores_file, "w") as f:
    json.dump({"accuracy": gb_test_acc}, f, indent=4)

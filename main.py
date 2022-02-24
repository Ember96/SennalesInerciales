import tsfel
import pandas as p
from os import walk
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from numpy import mean
from numpy import std

ds = p.DataFrame()

# def readData(path, caida):
#     try:
#         df = p.read_csv(path)
#         df = df.dropna()

#         cfg = tsfel.get_features_by_domain()

#         accel_x = tsfel.time_series_features_extractor(cfg, df['ax'], windows_size=100)
#         accel_y = tsfel.time_series_features_extractor(cfg, df['ay'], windows_size=100)
#         accel_z = tsfel.time_series_features_extractor(cfg, df['az'], windows_size=100)

#         giro_x = tsfel.time_series_features_extractor(cfg, df['wx'], windows_size=100)
#         giro_y = tsfel.time_series_features_extractor(cfg, df['wy'], windows_size=100)
#         giro_z = tsfel.time_series_features_extractor(cfg, df['wz'], windows_size=100)

#         otroDataFrame = p.DataFrame({"Fall": [caida]*accel_x.shape[0]})

#         return p.concat([otroDataFrame, accel_x, accel_y, accel_z, giro_x, giro_y, giro_z], axis=1)
#     except:
#         pass


# pathsFall = []
# for (dirpath, dirnames, filenames) in walk('Caida corrected/'):
#     pathsFall.extend(filenames)

# print(pathsFall)

# for x in pathsFall:
#     ds = p.concat([ds, readData('Caida corrected/' + x, 1)], axis=0)
#     print(readData('Caida corrected/' + x, 1))
# print(ds)

# for x in pathsFall:
#     ds = p.concat([ds, readData('No caida corrected/' + x, 0)], axis=0)
#     print(readData('No caida corrected/' + x, 1))
# print(ds)

# ds.to_csv(r'ds.csv', index = False)

ds = p.read_csv('ds.csv')

train = ds.loc[:, ds.columns != "Fall"] 
check = ds["Fall"]

cv = KFold(n_splits=12, random_state=1, shuffle=True)
model = tree.DecisionTreeClassifier()
scores = cross_val_score(model, train, check, scoring='accuracy', cv=cv, n_jobs=-1)
print('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))
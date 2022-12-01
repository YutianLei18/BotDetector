import pandas as pd
import numpy as np
file='/Users/apple/Desktop/IS & Senior Seminar/Research/机器人账号'

action=pd.read_csv(file+"train_actions_info.csv")

#log times:
log_times=action.groupby("user_id").agg({"action_id":"count"}).rename({"action_id":"cnt_act"})

#total pv:
pv_total=action.groupby("user_id").agg({"pv":"sum"}).rename({"pv":"sum_pv"})

#max,median,mean pv:
max_pv=action.groupby("user_id").agg({"pv":"max"}).rename({"pv":"max_pv"})

median_pv=action.groupby("user_id").agg({"pv":np.median}).rename({"pv":"median_pv"})

mean_pv=action.groupby("user_id").agg({"pv":"mean"}).rename({"pv":"mean_pv"})

#scale new features:
median_ratio_max_pv=action.groupby("user_id").agg\
    ({"pv":lambda x: np.median(x)/np.max(x)}).rename({"pv":"median_ratio_max_pv"})

mean_offset_hour=action.groupby("user_id").agg({"offset_hour":"mean"})\
    .rename({"offset_hour":"mean_offset_hour"})

truth=pd.read_csv(file+"train_ground_truth.csv")


from sklearn.ensemble import GradientBoostingClassifier as xgb
model=xgb(max_depth=4)

X=pd.concat([log_times,median_pv,mean_pv,max_pv,median_ratio_max_pv,pv_total,mean_offset_hour],axis=1)
X.reset_index(inplace=True)
y=X.join(truth.set_index('user_id'),on='user_id',how='inner')
label = y['label']
model.fit(X,label)

(label==model.predict(X)).mean()



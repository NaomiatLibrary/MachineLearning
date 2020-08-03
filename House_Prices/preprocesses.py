#前処理を行う関数を定義
from sklearn.preprocessing import StandardScaler
import re
import numpy as np

train_valiable_list=["LotFrontage","LotArea","OverallQual","OverallCond","YearBuilt","YearRemodAdd",
                "MasVnrArea","ExterQual","ExterCond","BsmtQual","BsmtCond","BsmtExposure",
            "BsmtFinType1","BsmtFinSF1","BsmtUnfSF","TotalBsmtSF","HeatingQC","CentralAir","1stFlrSF",
            "2ndFlrSF","LowQualFinSF","GrLivArea","BsmtFullBath","BsmtHalfBath","FullBath", "HalfBath",
            "BedroomAbvGr", "KitchenAbvGr",
            "TotRmsAbvGrd","KitchenQual","Fireplaces","FireplaceQu","GarageYrBlt",
            "GarageFinish","GarageCars", "GarageArea","GarageQual","GarageCond","PavedDrive",
            "WoodDeckSF", "OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch", "PoolArea",
            "PoolQC","MiscVal","YrSold","SalePrice"]
test_valiable_list=["LotFrontage","LotArea","OverallQual","OverallCond","YearBuilt","YearRemodAdd",
                "MasVnrArea","ExterQual","ExterCond","BsmtQual","BsmtCond","BsmtExposure",
            "BsmtFinType1","BsmtFinSF1","BsmtUnfSF","TotalBsmtSF","HeatingQC","CentralAir","1stFlrSF",
            "2ndFlrSF","LowQualFinSF","GrLivArea","BsmtFullBath","BsmtHalfBath","FullBath", "HalfBath",
            "BedroomAbvGr", "KitchenAbvGr",
            "TotRmsAbvGrd","KitchenQual","Fireplaces","FireplaceQu","GarageYrBlt",
            "GarageFinish","GarageCars", "GarageArea","GarageQual","GarageCond","PavedDrive",
            "WoodDeckSF", "OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch", "PoolArea",
            "PoolQC","MiscVal","YrSold"]
new_train_valiable_list=["LotFrontage","LotArea","OverallQual","OverallCond","YearBuilt","YearRemodAdd","MasVnrArea","ExterQual","ExterCond","BsmtQual","BsmtCond","BsmtExposure","BsmtFinType1","BsmtFinSF1","BsmtUnfSF","TotalBsmtSF","HeatingQC","CentralAir","1stFlrSF","2ndFlrSF","LowQualFinSF","GrLivArea","Baths","BedroomAbvGr", "KitchenAbvGr", "TotRmsAbvGrd","KitchenQual","Fireplaces","FireplaceQu","GarageYrBlt","GarageFinish","GarageCars", "GarageArea","GarageQual","GarageCond","PavedDrive","WoodDeckSF", "OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch", "PoolArea","PoolQC","MiscVal","YrSold","SalePrice"]
def ifnan(x):
    if isinstance(x, float) and x is np.nan:
        return 1
    return 0
def five_map1(x):
    if ifnan(x):
        return 0
    return {"Ex":5,"Gd":4,"TA":3,"Fa":2,"Po":1,"Na":0}[x]
def five_map2(x):
    if ifnan(x):
        return 0
    return {"Gd":4,"Av":3,"Mn":2,"No":1,"Na":0}[x]
def six_map1(x):
    if ifnan(x):
        return 0
    return {"GLQ":6,"ALQ":5,"BLQ":4,"Rec":3,"LwQ":2,"Unf":1,"NA":0}[x]
def YN_map(x):
    if ifnan(x):
        return 0
    return {"Y":1,"N":0}[x]
def three_map1(x):
    if ifnan(x):
        return 0
    return {"Fin":3,"RFn":2,"Unf":1,"NA":0}[x]
def YPN_map(x):
    if ifnan(x):
        return 0
    return {"Y":2,"P":1,"N":0}[x]
def preprocess(t):
    #n段階評価をmap
    for k in ["ExterQual","ExterCond","BsmtQual","BsmtCond","HeatingQC","KitchenQual","FireplaceQu","GarageQual","GarageCond","PoolQC"]:
        t[k]= t[k].map(five_map1)
    t["BsmtExposure"]= t["BsmtExposure"].map(five_map2)
    t["BsmtFinType1"]= t["BsmtFinType1"].map(six_map1)
    t["CentralAir"]= t["CentralAir"].map(YN_map)
    t["GarageFinish"]= t["GarageFinish"].map(three_map1)
    t["PavedDrive"]= t["PavedDrive"].map(YPN_map)
    #割合にする
    t["BsmtUnfSF"]=t["BsmtUnfSF"]/t["TotalBsmtSF"]
    #足し合わせる
    t["Baths"]=t["BsmtFullBath"]+t["BsmtHalfBath"]+ t["FullBath"]+t["HalfBath"]
    #取り除く
    t=t.drop(["BsmtFullBath","BsmtHalfBath","FullBath","HalfBath"], axis=1)
    #欠損データを最頻値で埋める
    t=t.fillna(t.mean())
    #Price以外の数値データをとりあえず標準化
    #collist=[ ]
    #t[collist]=t[collist].apply(lambda x : ((x - x.mean())*1/x.std()+0),axis=1)
    return t
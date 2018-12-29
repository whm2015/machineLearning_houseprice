# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
import numpy as np
from xgboost import plot_importance
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet, BayesianRidge
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.base import clone, BaseEstimator, TransformerMixin, RegressorMixin
from sklearn.preprocessing import Imputer, LabelEncoder, StandardScaler, RobustScaler
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold
from collections import defaultdict
from math import sqrt, log
import sys

pd.set_option('display.max_rows',None)

class DataFrameImputer(TransformerMixin):

    def __init__(self):
        """Impute missing values.

        Columns of dtype object are imputed with the most frequent value
        in column.

        Columns of other types are imputed with mean of column.

        """
    def fit(self, X, y=None):

        self.fill = pd.Series([X[c].value_counts().index[0]
            if X[c].dtype == np.dtype('O') else X[c].mean() for c in X],
            index=X.columns)

        return self

    def transform(self, X, y=None):
        return X.fillna(self.fill)


def map_values(df):
    df['MSSubClass'] = df['MSSubClass'].map({'180':1,
                                             '30':2, '45':2,
                                             '190':3, '50':3, '90':3, 
                                             '85':4, '40':4, '160':4, 
                                             '70':5, '20':5, '75':5, '80':5, '150':5,
                                             '120': 6, '60':6})
    df['MSZoning'] = df['MSZoning'].map({'C (all)':1, 'RM':2, 'RH':2, 'RL':3, 'FV':4})
    df['Neighborhood'] = df['Neighborhood'].map({'MeadowV':1,
                                               'IDOTRR':2, 'BrDale':2,
                                               'OldTown':3, 'Edwards':3, 'BrkSide':3,
                                               'Sawyer':4, 'Blueste':4, 'SWISU':4, 'NAmes':4,
                                               'NPkVill':5, 'Mitchel':5,
                                               'SawyerW':6, 'Gilbert':6, 'NWAmes':6,
                                               'Blmngtn':7, 'CollgCr':7, 'ClearCr':7, 'Crawfor':7,
                                               'Veenker':8, 'Somerst':8, 'Timber':8,
                                               'StoneBr':9,
                                               'NoRidge':10, 'NridgHt':10})
    df['HouseStyle'] = df['HouseStyle'].map({'1.5Unf':1, 
                                           '1.5Fin':2, '2.5Unf':2, 'SFoyer':2, 
                                           '1Story':3, 'SLvl':3,
                                           '2Story':4, '2.5Fin':4})
    df['MasVnrType'] = df['MasVnrType'].map({'BrkCmn':1, 'None':1, 'not':2, 'BrkFace':2, 'Stone':3})
    df['ExterQual'] = df['ExterQual'].map({'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5})
    df['ExterCond'] = df['ExterCond'].map({'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5})
    df['Foundation'] = df['Foundation'].map({'Slab':1, 'BrkTil':2, 'CBlock':2, 'Stone':2, 'Wood':3, 'PConc':4})
    df['BsmtQual'] = df['BsmtQual'].map({'not':1, 'Po':2, 'Fa':3, 'TA':4, 'Gd':5, 'Ex':6})
    df['BsmtCond'] = df['BsmtCond'].map({'not':1, 'Po':2, 'Fa':3, 'TA':4, 'Gd':5, 'Ex':6})
    df['BsmtExposure'] = df['BsmtExposure'].map({'not':1, 'No':2, 'Mn':3, 'Av':4, 'Gd':5})
    df['BsmtFinType1'] = df['BsmtFinType1'].map({'not':1, 'Unf':2, 'LwQ':3, 'Rec':4, 'BLQ':5, 'ALQ':6, 'GLQ':7})
    df['BsmtFinType2'] = df['BsmtFinType2'].map({'not':1, 'Unf':2, 'LwQ':3, 'Rec':4, 'BLQ':5, 'ALQ':6, 'GLQ':7})
    df['HeatingQC'] = df['HeatingQC'].map({'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5})
    df['KitchenQual'] = df['KitchenQual'].map({'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5})
    df['Functional'] = df['Functional'].map({'Maj2':1, 'Maj1':2, 'Min1':2, 'Min2':2, 'Mod':2, 'Sev':2, 'Typ':3})
    df['FireplaceQu'] = df['FireplaceQu'].map({'not':1, 'Po':2, 'Fa':3, 'TA':4, 'Gd':5, 'Ex':6})
    df['GarageFinish'] = df['GarageFinish'].map({'not':1, 'Unf':2, 'RFn':3, 'Fin':4})
    df['GarageQual'] = df['GarageQual'].map({'not':1, 'Po':2, 'Fa':3, 'TA':4, 'Gd':5, 'Ex':6})
    df['GarageCond'] = df['GarageCond'].map({'not':1, 'Po':2, 'Fa':3, 'TA':4, 'Gd':5, 'Ex':6})
    df['PavedDrive'] = df['PavedDrive'].map({'N':1, 'P':2, 'Y':3})
    df['PoolQC'] = df['PoolQC'].map({'not':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5})
    df['Fence'] = df['Fence'].map({'not':1, 'MnWw':2, 'GdWo':3, 'MnPrv':4, 'GdPrv':5})


def map_values2(df):
    df['Alley'] = df['Alley'].map({'not':1, 'Grvl':1, 'Pave':2, 'not':2})
    df['LotShape'] = df['LotShape'].map({'Reg':1, 'IR1':2, 'IR3':2, 'IR2':2})
    df['LandContour'] = df['LandContour'].map({'Bnk':1, 'Lvl':2, 'Low':3, 'HLS':4})
    df['LotConfig'] = df['LotConfig'].map({'Inside':1, 'Corner':1, 'FR2':1, 'CulDSac':2, 'FR3':2})
    df['LandSlope'] = df['LandSlope'].map({'Gtl':1, 'Sev':2, 'Mod':3})
    df['Condition1'] = df['Condition1'].map({'Artery':1, 'Feedr':2, 'RRAe':2, 'Norm':3, 'RRAn':3, 'PosA':4, 'RRNe':4, 'PosN':4, 'RRNn':5})
    df['Condition2'] = df['Condition2'].map({'Artery':1, 'RRNn':2, 'Feedr':2, 'Norm':3, 'RRAn':3, 'PosA':4, 'RRAe':4, 'PosN':4})
    df['BldgType'] = df['BldgType'].map({'2fmCon':1, 'Duplex':1, 'Twnhs':1, '1Fam':2, 'TwnhsE':2})
    df['RoofStyle'] = df['RoofStyle'].map({'Gambrel':1, 'Gable':2, 'Flat':2, 'Hip':2, 'Mansard':3, 'Shed':3})
    df['RoofMatl'] = df['RoofMatl'].map({'ClyTile':1, 'Roll':1, 'CompShg':1, 'Tar&Grv':1, 'Metal':1, 'WdShake':2, 'WdShngl':2})
    df['Heating'] = df['Heating'].map({'Floor':1, 'Grav':1, 'Wall':2, 'OthW':3, 'GasW':4, 'GasA':4})
    df['CentralAir'] = df['CentralAir'].map({'N':1, 'Y':2})
    df['Electrical'] = df['Electrical'].map({'not':1, 'Mix':1, 'FuseP':1, 'FuseA':2, 'FuseF':2, 'SBrkr':3})
    df['GarageType'] = df['GarageType'].map({'not':1, 'CarPort':2, 'Detchd':3, 'Basment':4, '2Types':4, 'Attchd':5, 'BuiltIn':5})
    df['MiscFeature'] = df['MiscFeature'].map({'not':1, 'Shed':2, 'Gar2':3, 'Othr':3, 'Othr':4})
    df['SaleType'] = df['SaleType'].map({'Oth':1, 'ConLw':2, 'ConLD':2, 'COD':2, 'WD':3, 'ConLI':3, 'CWD':4, 'New':4, 'Con':5})
    df['SaleCondition'] = df['SaleCondition'].map({'AdjLand':1, 'Abnorml':2, 'Family':2, 'Alloca':2, 'Normal':3, 'Partial':4})


def add_feature(X):
    X['TotalHouse'] = X['TotalBsmtSF'] + X['1stFlrSF'] + X['2ndFlrSF']
    X['TotalArea'] = X['TotalBsmtSF'] + X['1stFlrSF'] + X['2ndFlrSF'] + X['GarageArea']
    X['TotalQuality'] = X['OverallQual'] + X['OverallCond']
    X['TotalHouse_OverallQual'] = X['TotalHouse'] * X['OverallQual']
    X['GrLivArea_OverallQual'] = X['GrLivArea'] * X['OverallQual']
    X['MSZoning_TotalHouse'] = X['MSZoning'] * X['TotalHouse']
    X['MSZoning_OverallQual'] = X['MSZoning'] + X['OverallQual']
    X['MSZoning_YearBuilt'] = X['MSZoning'] + X['YearBuilt']
    X['Neighborhood_TotalHouse'] = X['Neighborhood'] * X['TotalHouse']
    X['Neighborhood_OverallQual'] = X['Neighborhood'] + X['OverallQual']
    X['Neighborhood_YearBuilt'] = X['Neighborhood'] + X['YearBuilt']
    X['BsmtFinSF1_OverallQual'] = X['BsmtFinSF1'] * X['OverallQual']
    X['Functional_TotalHouse'] = X['Functional'] * X['TotalHouse']
    X['Functional_OverallQual'] = X['Functional'] + X['OverallQual']
    X['LotArea_OverallQual'] = X['LotArea'] * X['OverallQual']
    X['LotArea_TotalHouse'] = X['LotArea'] + X['TotalHouse']
    X['Bsmt'] = X['BsmtFinSF1'] + X['BsmtFinSF2'] + X['BsmtUnfSF']
    X['PorchArea'] = X['OpenPorchSF'] + X['EnclosedPorch'] + X['3SsnPorch'] + X['ScreenPorch']
    X['TotalPlace'] = X['TotalArea'] + X['PorchArea']


added_features = ['TotalHouse', 'TotalArea', 'TotalQuality', 'TotalHouse_OverallQual',
                  'GrLivArea_OverallQual', 'MSZoning_TotalHouse', 'MSZoning_OverallQual',
                  'MSZoning_YearBuilt', 'Neighborhood_TotalHouse', 'Neighborhood_OverallQual',
                  'Neighborhood_YearBuilt', 'BsmtFinSF1_OverallQual', 'Functional_TotalHouse',
                  'Functional_OverallQual', 'LotArea_OverallQual', 'LotArea_TotalHouse',
                  'Bsmt', 'PorchArea', 'TotalPlace']


def loadDataset(trainfile, testfile):
    train_df = pd.read_csv(trainfile)
    train_X = train_df.iloc[:, 1:-1]              #去掉id和price
    train_y = train_df.SalePrice
    test_df = pd.read_csv(testfile)
    test_X = test_df.iloc[:, 1:]
    testindex = test_df[['Id']]
   
    train_number = train_X.shape[0]
    full = pd.concat([train_X, test_X], axis=0, ignore_index=True)

    Num_to_Str = ['MSSubClass']
    nan_means_none = ['Alley', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
                      'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
                      'PoolQC', 'Fence', 'MiscFeature', 'MasVnrType'
                     ]
    nan_especial = ['Electrical']
    nan_means_zero = ['MasVnrArea']
    
    full[nan_means_none] = full[nan_means_none].fillna('not')
    full[nan_especial] = full[nan_especial].fillna('not')
    full[nan_means_zero] = full[nan_means_zero].fillna(0)
    for col in Num_to_Str:
        full[col] = full[col].astype(str)
    
    nan_count = full.isnull().sum()
    print(nan_count[nan_count>0])
    
    map_values(full)
    #map_values2(full)
    
    print(full.select_dtypes(include='object').columns)
    #les = defaultdict(LabelEncoder)
    imputer = Imputer(missing_values=np.nan, strategy='mean', axis=0)
    std = StandardScaler()
    rbX = RobustScaler()

    add_feature(full)

    '''
    full_numeric = full.select_dtypes(exclude='object')
    skew = full_numeric.apply(lambda x: x.skew())
    skew_features = skew[abs(skew) >= 1].index
    print(full[skew_features].dtypes)
    full[skew_features] = np.log1p(full[skew_features])
    '''
    
    nan_count = full.isnull().sum()
    nan_index = nan_count[nan_count>0].index
    print(nan_index)
    full[nan_index]= imputer.fit_transform(full[nan_index])

    full = pd.get_dummies(full)
    print(full.shape)
    
    #pca = PCA(n_components=120)
    #full = pca.fit_transform(full)

    '''
    if 'O' in full.dtypes.tolist():
        for i in full:
            if full[i].dtype.kind == 'O':
                full[i] = LabelEncoder().fit_transform(full[i])
    '''

    #datatype_group = train_X.columns.to_series().groupby(train_X.dtypes).groups
    #datatype_group = {k.name: v.tolist() for k, v in datatype_group.items()}
    #object_columns_name = datatype_groupobject']
    #digit_columns_name = datatype_group['int64'] + datatype_group['float64']
    #full[object_columns_name] = full[object_columns_name].apply(lambda x: les[x.name].transform(x))
    
    #test_X[object_columns_name] = test_X[object_columns_name].apply(lambda x: les[x.name].transform(x))
    #test_X[digit_columns_name] = imputer.transform(test_X[digit_columns_name])

    full = std.fit_transform(full)
    
    train_y = np.log(train_y)

    train_X = full[:train_number]
    test_X = full[train_number:]
    print(type(full))
    return train_X, train_y, test_X, testindex


def trainandTest(X_train, y_train, X_test, testindex, output_path):
    rbY = RobustScaler()
    y_train = rbY.fit_transform(y_train.values.reshape(-1,1))
    model = xgb.XGBRegressor(n_estimators=210,
                             subsample=0.7, max_depth=3, min_child_weight=1, seed=0,
                             colsample_bytree=0.8,
                             #learning_rate=0.21, gamma=0.14, reg_alpha=0.015, reg_lambda=0.002,
                             silent=1, objective='reg:linear')
    model.fit(X_train, y_train)
    ans = model.predict(X_test)
    ans = rbY.inverse_transform(ans.reshape(-1,1))
    ans = np.exp(ans)
    result = testindex
    result['SalePrice'] = ans
    result.to_csv(output_path, index=None)

    # 显示重要特征
    # plot_importance(model)
    # plt.show()

    bns = np.exp(rbY.inverse_transform(model.predict(X_train).reshape(-1,1)))
    error = []
    for i ,j in zip(bns, y_train):
        error.append(abs(i-j))
    print(bns.tolist())
    print('xun训练集误差')
    print(sqrt(sum(error)/len(error)))


def trainandTestLR(X_train, y_train, X_test, testindex, output_path):
    rbY = RobustScaler()
    y_train = rbY.fit_transform(y_train.values.reshape(-1,1))
    model = LinearRegression()
    model.fit(X_train, y_train)
    ans = model.predict(X_test)
    ans = rbY.inverse_transform(ans.reshape(-1,1))
    ans = np.exp(ans)
    result = testindex
    result['SalePrice'] = ans
    result.to_csv(output_path, index=None)


def trainandTestSVR(X_train, y_train, X_test, testindex, output_path):
    rbY = RobustScaler()
    y_train = rbY.fit_transform(y_train.values.reshape(-1,1))
    model = SVR(gamma='auto', kernel='rbf')
    model.fit(X_train, y_train)
    ans = model.predict(X_test)
    ans = rbY.inverse_transform(ans.reshape(-1,1))
    ans = np.exp(ans)
    result = testindex
    result['SalePrice'] = ans
    result.to_csv(output_path, index=None)

    bns = np.exp(rbY.inverse_transform(model.predict(X_train).reshape(-1,1)))
    error = []
    for i ,j in zip(bns, y_train):
        error.append(abs(i-j))
    print(bns.tolist())
    print('xun训练集误差')
    print(sqrt(sum(error)/len(error)))


def trainandTestKR(X_train, y_train, X_test, testindex, output_path):
    rbY = RobustScaler()
    y_train = rbY.fit_transform(y_train.values.reshape(-1,1))
    model = KernelRidge(alpha=0.19, kernel='laplacian', coef0=0)
    model.fit(X_train, y_train)
    ans = model.predict(X_test)
    ans = rbY.inverse_transform(ans.reshape(-1,1))
    ans = np.exp(ans)
    result = testindex
    result['SalePrice'] = ans
    result.to_csv(output_path, index=None)

    bns = np.exp(rbY.inverse_transform(model.predict(X_train).reshape(-1,1)))
    error = []
    for i ,j in zip(bns, y_train):
        error.append(abs(i-j))
    print(bns.tolist())
    print('xun训练集误差')
    print(sqrt(sum(error)/len(error)))


def trainandTestLa(X_train, y_train, X_test, testindex, output_path):
    print(type(X_train))
    rbY = RobustScaler()
    y_train = rbY.fit_transform(y_train.values.reshape(-1,1))
    model = Lasso(alpha=0.06, max_iter=2000, selection='random', tol=0.001, normalize=False)
    model.fit(X_train, y_train)
    print(type(X_train))
    coef = pd.Series(model.coef_, index = X_train.columns)# .coef_ 可以返回经过学习后的所有 feature 的参数。
    print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")
    coef_all = coef[coef!=0].sort_values()
    print(set(coef_all.index.tolist()) & set(added_features))
    print(coef_all.tail(20))
    print(coef_all.head(20))
    ans = model.predict(X_test)
    ans = rbY.inverse_transform(ans.reshape(-1,1))
    ans = np.exp(ans)
    result = testindex
    result['SalePrice'] = ans
    result.to_csv(output_path, index=None)

    bns = np.exp(rbY.inverse_transform(model.predict(X_train).reshape(-1,1)))
    error = []
    for i ,j in zip(bns, y_train):
        error.append(abs(i-j))
    print(bns.tolist())
    print('xun训练集误差')
    print(sqrt(sum(error)/len(error)))


def grid(model, X, y, params):
    rbY = RobustScaler()
    y = rbY.fit_transform(y.values.reshape(-1,1))
    grid_search = GridSearchCV(model, params, scoring='neg_mean_squared_error', cv=5, verbose=1, n_jobs=30)
    grid_search.fit(X, y.ravel())
    #print('每轮迭代运行结果:{0}'.format(grid_search.evalute_result))
    print('参数的最佳取值：{0}'.format(grid_search.best_params_))
    print('最佳模型得分:{0}'.format(grid_search.best_score_))


def grid_searchX(X_train, y_train):
    cv_params = {'n_estimators': [210, ], 
                 'subsample': [0.7, ],
                 'colsample_bytree': [0.8, ],
                 'max_depth': [3, ], 
                 'min_child_weight': [1, ],
                 'learning_rate': [0.21, ], 
                 'gamma': [0.14, ], 
                 'reg_alpha': [0.015, ], 
                 'reg_lambda': [0.002, ]
                 }
    other_params = {'seed': 0,
                    }
    model = xgb.XGBRegressor(**other_params)
    grid(model, X_train, y_train, cv_params)


def grid_searchKR(X_train, y_train):
    cv_params = {'alpha': [0.1 + x/100 for x in range(0, 20)],
                 'kernel': ['laplacian', 'cosine'],
                 'coef0' : [0, 0.1, 0.01, 0.001]
                 }
    model = KernelRidge()
    grid(model, X_train, y_train, cv_params)


def grid_searchSVR(X_train, y_train):
    cv_params = {'C': range(5, 15),
                 'kernel': ['poly', 'rbf'],
                 'degree': [2,3,4],
                 'epsilon': [x/100 for x in range(1,10)]
                }
    model = SVR(gamma='auto')
    grid(model, X_train, y_train, cv_params)


def grid_searchBR(X_train, y_train):
    cv_params = {'n_iter': range(100, 1000, 100),#100
                 'tol': [1e-4, 1e-5],#1e-5
                 'alpha_1': [1e-7, 1e-8],#1e-8
                 'alpha_2': [1e-4, 1e-5],#1e-4
                 'lambda_1': [1e-4, 1e-5],#1e-4
                 'lambda_2': [1e-7, 1e-8],#1e-8
                }
    model = BayesianRidge(normalize=False)
    grid(model, X_train, y_train, cv_params)


def grid_searchLa(X_train, y_train):
    cv_params = {'alpha': [0.06],
                 'max_iter': range(1000, 3000, 100),
                 'selection': ['random'],
                 'tol': [1e-2, 1e-3, 1e-4]
                }
    model = Lasso(normalize=False)
    grid(model, X_train, y_train, cv_params)


def grid_searchEN(X_train, y_train):
    cv_params = {'alpha': [1],#0.1
                 'l1_ratio': [x/10 for x in range(11)],#0.2
                 'max_iter': [10000],#2800
                 'selection': ['random'],
                 #'tol': [1e-2, 1e-3]#1e-2
                }
    model = ElasticNet(normalize=False)
    grid(model, X_train, y_train, cv_params)


def calculate_error(output_path, real_results):
    test = pd.read_csv(output_path)
    real = pd.read_csv(real_results)
    real = real.loc[1160:, ['Id', 'SalePrice']]
    test_value = test['SalePrice'].values.tolist()
    real_value = real['SalePrice'].values.tolist()
    print('test_values')
    print(test_value)
    print('real_values')
    print(real_value)
    error = []
    for i, j in zip(test_value, real_value):
        error.append(abs(i-j))
    print('error')
    print(error)
    print('误差和')
    print(sum(error))
    print(len(test_value))
    print(sum(error)/len(test_value))
    print(sqrt(sum(error)/len(test_value)))


def rmse_cv(model, X, y):
    rmse = np.sqrt(-cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=5))
    return rmse


class AverageWeight(BaseEstimator, RegressorMixin):
    def __init__(self, mods, weight):
        self.mods = mods
        self.weight = weight

    def fit(self, X, y):
        self.models_ = [clone(x) for x in self.mods]
        for model in self.models_:
            model.fit(X, y)
        return self

    def predict(self, X):
        results = [model.predict(X) for model in self.models_]
        # 各个模型预测结果加权平均
        pre = np.dot(np.array(self.weight), np.array(results))
        return pre


class stacking(BaseEstimator, RegressorMixin):
    def __init__(self, base_models, stack_model):
        self.base_models = base_models
        self.stack_model = stack_model
        self.kf = KFold(n_splits=5, random_state=42, shuffle=True)

    def fit(self, X, y):
        # 注：这里要把数据转换成数组类型，避免传入的数据不是数组类型时报错
        X = np.array(X)
        y = np.array(y)

        self.saved_models = [list() for model in self.base_models]
        oof_train = np.zeros((X.shape[0], len(self.base_models)))
        for i, model in enumerate(self.base_models):
            for train_idx, val_idx in self.kf.split(X, y):
                fit_model = clone(model)            
                fit_model.fit(X[train_idx], y[train_idx])
                self.saved_models[i].append(fit_model)
                oof_train[val_idx, i] = fit_model.predict(X[val_idx])

        self.stack_model.fit(oof_train, y)
        return self

    def predict(self, X):
        X = np.array(X)
        oof_test = np.zeros((X.shape[0], len(self.base_models)))

        for i, model in enumerate(self.saved_models):
            model_pred = np.column_stack([fit_model.predict(X) for fit_model in model])
            oof_test[:, i] = model_pred.mean(1)

        return self.stack_model.predict(oof_test)


if __name__ == '__main__':
    trainFilePath = r'houseprice_train.csv'
    testFilePath = r'houseprice_test.csv'
    output_path = r'submission.csv'
    real_results = r'train.csv'
    X_train, y_train, X_test, testindex = loadDataset(trainFilePath, testFilePath)
    print(y_train.shape)
    trainandTestKR(X_train, y_train, X_test, testindex, output_path)
    calculate_error(output_path, real_results)
    #grid_searchX(X_train, y_train)
'''
    rbY = RobustScaler()
    y_train = rbY.fit_transform(y_train.values.reshape(-1,1)).ravel()
    xgb = xgb.XGBRegressor(n_estimators=250, subsample=0.7, max_depth=4, min_child_weight=1,
                           seed=0, colsample_bytree=0.8,
                           #learning_rate=0.07, gamma=0, reg_alpha=0, reg_lambda=1,
                           silent=1, objective='reg:linear')
    kr = KernelRidge(alpha=0.19, kernel='laplacian', coef0=0)
    en = ElasticNet(alpha=0.1, l1_ratio=0.2, max_iter=2800, selection='random', tol=1e-2, normalize=False)
    la = Lasso(alpha=0.06, max_iter=2000, selection='random', tol=0.001, normalize=False)
    svr = SVR(gamma='auto', kernel='rbf')
    br = BayesianRidge(n_iter=100, tol=1e-5, alpha_1=1e-8, alpha_2=1e-4, lambda_1=1e-4, lambda_2=1e-8, normalize=False) 
    #model_weight = [0.5, 0.5, 0]
    #models = [xgb, kr]
    #avg_w = AverageWeight(models, model_weight)

    # 交叉验证评估
    #avg_score = rmse_cv(avg_w, X_train, y_train)
    #print(avg_score.mean())

    stack_model = stacking(base_models=[xgb, kr], stack_model=kr)
    stack_score = rmse_cv(stack_model, X_train, y_train)
    print(stack_score.mean())

    stack_model.fit(X_train, y_train)
    stack_pred = np.exp(rbY.inverse_transform(stack_model.predict(X_test).reshape(-1,1)))
    result_s = testindex
    result_s['SalePrice'] = stack_pred
    result_s.to_csv(output_path, index=None)
    calculate_error(output_path, real_results)
'''

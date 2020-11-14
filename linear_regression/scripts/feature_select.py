import pandas as pd
from sklearn.model_selection import train_test_split


class Train:
    '''
    Class Train contains methods to perform data cleaning of a particular
    DataFrame for the Ames Housing data set.
    '''
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.cleaned = self.df.copy()

    def show_stats(self, series: str):
        """Print stats from a Pandas Series"""

            # Describe
        descriptive_stats = self.cleaned[series].describe()
        print('Describe Method:')
        print(descriptive_stats)
        print('---------------------------------------------------')
        
        # Numeric Columns only
        if self.cleaned[series].dtype == 'int64':
            mean = self.cleaned[series].mean()
            maximum = max(self.cleaned[series])
            minimum = min(self.cleaned[series])
            print(f'''
The Min/Max of column {self.cleaned[series].name}: ({minimum}, {maximum})
        
The maximum is {maximum - mean} away from the mean
The minimum is {mean - minimum} away from the mean
            ''')
            print('---------------------------------------------------')
        elif self.cleaned[series].dtype == 'O':
            mode = self.cleaned[series].mode()
            print(f'The most freqent class is: {mode[0]}')
            print('---------------------------------------------------')
            
            print(f'Value Counts for column: {self.cleaned[series].name}')
            print(self.cleaned[series].value_counts())
            print('---------------------------------------------------')
        
        # Number of NaN values
        nans = self.cleaned[series].isna().sum()
        nans_to_percent = nans/len(self.cleaned[series] * 100)
        print(f'Number of NaNs: {nans}')
        print(f'Percent of Null Values for the column: {nans_to_percent}%')
        print('---------------------------------------------------')
        
        # Number of Unique Values - Only display if there are less than 20 unique
        unique = self.cleaned[series].unique()
        print(f'There are {len(unique)} values')
        
        if len(unique) <= 20:
            print('Unique Values:')
            print(unique)
            print('---------------------------------------------------')
        else:
            print('Warning: High Cardinality')
            print('---------------------------------------------------')

    def train_val_test_split(self, X, y, random_state, test_size=0.20, val_size=0.25):
        '''
        Performs two train_test_splits on the dataset returning X and y for
        train, validation and test sets.

        Parameters:
            - X: Feature data as a Pandas.DataFrame object
            - y: Target column data as a Pandas.Series object
            - random_state: Int value applied to the random_state parameter
                            of both train_test_split calls.
            - test_size (default=0.20): Percentage (0-1) representing a
                        proportion of the (X, y) data PRIOR to first split.
            - val_size (default=0.25): Percentage (0-1) representing a
                        proportion of the (X, y) data POST first split.
        
        Returns (in order):
            - X_train, y_train, X_val, y_val, X_test, y_test
        '''

        X_remain, X_test, y_remain, y_test = train_test_split(X, y, test_size=test_size,
                                                              random_state=random_state)

        X_train, X_val, y_train, y_val = train_test_split(X_remain, y_remain, test_size=val_size,
                                                          random_state=random_state)

        return X_train, y_train, X_val, y_val, X_test, y_test

    def make_conditions(self):

        # 1. LotArea > 95-percentile (17401.15 ftsq)
        condition_LotArea = (self.cleaned['LotArea'] <= 17401.15)

        # 2. TotalBsmtSF > 1753
        condition_TotalBsmtSF = (self.cleaned['TotalBsmtSF'] <= 1753)

        # 3. 1stFlrSF > 1831.25
        condition_1stFlrSF = (self.cleaned['1stFlrSF'] <= 1831.25)

        # 4. GrLivArea > 2466.1
        condition_GrLivArea = (self.cleaned['GrLivArea'] <= 2466.1)

        # 5. SalePrice > 326100
        condition_SalePrice = (self.cleaned['SalePrice'] <= 326100)

        return [condition_LotArea, condition_TotalBsmtSF, condition_1stFlrSF, condition_GrLivArea, condition_SalePrice]

    def clean_df(self, condition_LotArea=None, condition_TotalBsmtSF=None, condition_1stFlrSF=None, condition_GrLivArea=None, condition_SalePrice=None):
        '''
        This cleans up the dataframe, makes a few features and ordinally encodes some categorical features.
        
        Returns a DataFrame for Train or Test set depending on whether conditionals were given - default is for Test
        '''

        # First, remove all NaN values
        condition0 = (len(self.cleaned) * 0.1)
        cols = self.cleaned.columns

        for col in cols:
            n_NaN = (self.cleaned[col].isna().sum())
            if n_NaN > condition0:
                self.cleaned.drop(columns=col, inplace=True)
            elif 0 < n_NaN < condition0:
                if self.cleaned[col].dtype != 'O':
                    if len(self.cleaned[col].unique()) > 20:
                        self.cleaned[col].fillna(value=self.cleaned[col].mean(), inplace=True)
                    else:
                        self.cleaned[col].fillna(value=self.cleaned[col].mode()[0], inplace=True)
                else:
                    self.cleaned[col].fillna(value=self.cleaned[col].mode()[0], inplace=True)

        # 1. LotShape - (Combine Irregular)
        self.cleaned.loc[self.cleaned['LotShape'].str.startswith('IR'), 'RegularLotShape'] = 0
        self.cleaned.loc[self.cleaned['LotShape'].str.startswith('Reg'), 'RegularLotShape'] = 1

        # 2. LandContour - (Combine non Lvl values)
        self.cleaned.loc[(self.cleaned['LandContour'] == 'Bnk') | (self.cleaned['LandContour'] == 'HLS') | (self.cleaned['LandContour'] == 'Low'), 'LandIsLvl'] = 0
        self.cleaned.loc[self.cleaned['LandContour'] == 'Lvl', 'LandIsLvl'] = 1

        # 3. LotConfig - (FR2, FR3 essentially the same)
        # Ordinality - {'Inside': 0, 'Corner': 1, 'CulDSac': 2, 'FR': 3}
        self.cleaned.loc[self.cleaned['LotConfig'] == 'Inside', 'LotConfigCL'] = 0
        self.cleaned.loc[self.cleaned['LotConfig'] == 'Corner', 'LotConfigCL'] = 1
        self.cleaned.loc[self.cleaned['LotConfig'] == 'CulDSac', 'LotConfigCL'] = 2
        self.cleaned.loc[self.cleaned['LotConfig'].str.startswith('FR'), 'LotConfigCL'] = 3

        # 4. Condition1 - (Combine adjacency types)
        # Ordinality - {'Norm': 0, 'Feedr/Artery': 1, 'RRA/N': 2, 'PosFeat': 3}
        self.cleaned.loc[self.cleaned['Condition1'] == 'Norm', 'LotAdjacencyType'] = 0
        self.cleaned.loc[(self.cleaned['Condition1'] == 'Feedr') | (self.cleaned['Condition1'] == 'Artery'), 'LotAdjacencyType'] = 1
        self.cleaned.loc[self.cleaned['Condition1'].str.startswith('RR'), 'LotAdjacencyType'] = 2
        self.cleaned.loc[self.cleaned['Condition1'].str.startswith('Pos'), 'LotAdjacencyType'] = 3

        # 5. OverallQual - (Combine extremes)
        # Ordinality - {'below_4': 0, 'Average(4,5,6)': 1, 'above_6': 2}
        self.cleaned.loc[self.cleaned['OverallQual'] < 4, 'HouseCondition'] = 0
        self.cleaned.loc[self.cleaned['OverallQual'] <= 6, 'HouseCondition'] = 1
        self.cleaned.loc[self.cleaned['OverallQual'] >= 7, 'HouseCondition'] = 2

        # 6. YearBuilt - Split {MadeBefore1946: 0, MadeAfter1946: 1}
        self.cleaned.loc[self.cleaned['YearBuilt'] < 1946, 'YrBuilt'] = 0
        self.cleaned.loc[self.cleaned['YearBuilt'] >= 1946, 'YrBuilt'] = 1

        # 7. YearRemodAdd - NEW COLUMN - WasRemodeled
        # Process - If the years for YearBuilt and YearRemodAdd are the same, there was no remodel
        self.cleaned.loc[self.cleaned['YearBuilt'] == self.cleaned['YearRemodAdd'], 'WasRemodeled'] = 0
        self.cleaned.loc[self.cleaned['YearBuilt'] != self.cleaned['YearRemodAdd'], 'WasRemodeled'] = 1

        # 8. MasVnrType - (Combine brick-types)
        # Ordinality - {'None': 0, 'Brick': 1, 'Stone': 3}
        self.cleaned.loc[self.cleaned['MasVnrType'] == 'None', 'VeneerType'] = 0
        self.cleaned.loc[self.cleaned['MasVnrType'].str.startswith('Brk'), 'VeneerType'] = 1
        self.cleaned.loc[self.cleaned['MasVnrType'] == 'Stone', 'VeneerType'] = 2

        # 9. HeatingQC - (Combine Fair and Poor - heating is important!)
        # Ordinality - {'Excellent': 0, 'Average': 1, 'Good': 2, 'Poor': 3}
        self.cleaned.loc[self.cleaned['HeatingQC'] == 'Ex', 'HeatingQuality'] = 0
        self.cleaned.loc[self.cleaned['HeatingQC'] == 'TA', 'HeatingQuality'] = 1
        self.cleaned.loc[self.cleaned['HeatingQC'] == 'Gd', 'HeatingQuality'] = 2
        self.cleaned.loc[(self.cleaned['HeatingQC'] == 'Fa') | (self.cleaned['HeatingQC'] == 'Po'), 'HeatingQuality'] = 3

        # 10. Electrical - (Combine all Fuse types)
        # Binary - {'Breaker': 0, 'Fuse': 1}
        self.cleaned.loc[self.cleaned['Electrical'] == 'SBrkr', 'EleSystem'] = 0
        self.cleaned.loc[(self.cleaned['Electrical'].str.startswith('Fuse')) | (self.cleaned['Electrical'] == 'Mix'), 'EleSystem'] = 1

        # 11. BsmtFull/HalfBath - NEW COLUMN - BsmtHasBath
        self.cleaned.loc[(self.cleaned['BsmtFullBath'] == 0) | (self.cleaned['BsmtHalfBath'] == 0), 'BsmtHasBath'] = 0
        self.cleaned.loc[(self.cleaned['BsmtFullBath'] > 0) | (self.cleaned['BsmtHalfBath'] > 0), 'BsmtHasBath'] = 1

        # 12. HalfBath - (Combine 1 and 2 to make binary) - HasHalfBath
        self.cleaned.loc[self.cleaned['HalfBath'] == 0, 'HasHalfBath'] = 0
        self.cleaned.loc[self.cleaned['HalfBath'] > 0, 'HasHalfBath'] = 1

        # 13. BedroomAbvGr - (0-1, 2, 3, 4+)
        # Ordinality - {'less_than_2': 0, '2': 1, '3': 2, '4+': 3}
        self.cleaned.loc[self.cleaned['BedroomAbvGr'] < 2, 'Bedrooms'] = 0
        self.cleaned.loc[self.cleaned['BedroomAbvGr'] == 2, 'Bedrooms'] = 1
        self.cleaned.loc[self.cleaned['BedroomAbvGr'] == 3, 'Bedrooms'] = 2
        self.cleaned.loc[self.cleaned['BedroomAbvGr'] > 3, 'Bedrooms'] = 3

        # 14. TotRmsAvbGrd - NEW COLUMN - AdditionalRooms
        # Make a new column called RemainingRooms that is the difference between Total Rooms and Bedrooms
        # Ordinality - {'less_than_3': 0, '3': 1, '4': 2, '5': 3, 'more_than_5': 4}
        self.cleaned['RemainingRooms'] = self.cleaned['TotRmsAbvGrd'] - self.cleaned['BedroomAbvGr']
        self.cleaned.loc[self.cleaned['RemainingRooms'] < 3, 'AdditionalRooms'] = 0
        self.cleaned.loc[self.cleaned['RemainingRooms'] == 3, 'AdditionalRooms'] = 1
        self.cleaned.loc[self.cleaned['RemainingRooms'] == 4, 'AdditionalRooms'] = 2
        self.cleaned.loc[self.cleaned['RemainingRooms'] == 5, 'AdditionalRooms'] = 3
        self.cleaned.loc[self.cleaned['RemainingRooms'] > 5, 'AdditionalRooms'] = 4

        # 15. Fireplaces - (Combine 2 and 3)
        # Ordinality - {'None': 0, '1': 1, '2+': 2}
        self.cleaned.loc[self.cleaned['Fireplaces'] == 0, 'NumFireplaces'] = 0
        self.cleaned.loc[self.cleaned['Fireplaces'] == 1, 'NumFireplaces'] = 1
        self.cleaned.loc[self.cleaned['Fireplaces'] > 1, 'NumFireplaces'] = 2

        # 16. GarageCars - (Combine 3 and 4)
        # Ordinality - {'0': 0, '1': 1, '2': 2, '3+': 3}
        self.cleaned.loc[self.cleaned['GarageCars'] == 0, 'GarageAreaByCar'] = 0
        self.cleaned.loc[self.cleaned['GarageCars'] == 1, 'GarageAreaByCar'] = 1
        self.cleaned.loc[self.cleaned['GarageCars'] == 2, 'GarageAreaByCar'] = 2
        self.cleaned.loc[self.cleaned['GarageCars'] > 2, 'GarageAreaByCar'] = 3

        # 17. WoodDeckSF - NEW COLUMN - HasDeck
        self.cleaned.loc[self.cleaned['WoodDeckSF'] == 0, 'HasDeck'] = 0
        self.cleaned.loc[self.cleaned['WoodDeckSF'] > 0, 'HasDeck'] = 1

        # 18. PoolArea - NEW COLUMN - HasPool
        self.cleaned.loc[self.cleaned['PoolArea'] == 0, 'HasPool'] = 0
        self.cleaned.loc[self.cleaned['PoolArea'] > 0, 'HasPool'] = 1

        # 19. MoSold - Subtract all items by 1
        # Ordinality - {'Jan': 0 ... 'Dec': 11}
        self.cleaned['MonthSold'] = self.cleaned['MoSold'] - 1

        # 20. YrSold - Convert years to 0-4 - 2010 might not have concluded at creation of dataset
        # Ordinality - {'2006': 0, '2007': 1, '2008': 2, '2009': 3, '2010': 4}
        self.cleaned.loc[self.cleaned['YrSold'] <= 2006, 'YearSold'] = 0
        self.cleaned.loc[self.cleaned['YrSold'] == 2007, 'YearSold'] = 1
        self.cleaned.loc[self.cleaned['YrSold'] == 2008, 'YearSold'] = 2
        self.cleaned.loc[self.cleaned['YrSold'] == 2009, 'YearSold'] = 3
        self.cleaned.loc[self.cleaned['YrSold'] == 2010, 'YearSold'] = 4
        
        # =====================================================================================

        new_columns = list(self.cleaned.columns[-21:])

        features_encoded = self.cleaned[new_columns].astype('int64')

        if condition_LotArea != None:

            df_inter = pd.concat([self.cleaned[['LotArea', 'TotalBsmtSF', '1stFlrSF', 'GrLivArea', 'SalePrice']], features_encoded], axis=1)

            # Final data set should be shape (1253, 26)

            final_clean_filtered = df_inter[condition_LotArea & condition_TotalBsmtSF & condition_1stFlrSF & condition_GrLivArea & condition_SalePrice]

            return final_clean_filtered  # Specifically for Train data set
        
        else:
            
            df_inter = pd.concat([self.cleaned[['Id', 'LotArea', 'TotalBsmtSF', '1stFlrSF', 'GrLivArea']], features_encoded], axis=1)

            return df_inter  # This is specifically for the Test data set (does not have SalePrice)

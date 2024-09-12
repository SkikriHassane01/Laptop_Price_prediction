# at first we need to add the feature creation to the start of our pipeline so let's do it:
# here is all the preprocessing that we did above:


class FeatureEngineeringTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        # Extract resolution and ScreenType
        X['resolution'] = X['ScreenResolution'].str.extract(r'(\d+x\d+)')
        X['ScreenType'] = X['ScreenResolution'].replace(r'(\d+x\d+)', '', regex=True)
        X['ScreenType'] = X['ScreenType'].replace(r'^\s*$', np.nan, regex=True)
        
        # Extract TouchScreen
        X['TouchScreen'] = X['ScreenType'].str.extract(r'(Touchscreen)').notna().astype(int)
        X['ScreenType'] = X['ScreenType'].str.replace(r'(\/\sTouchscreen)', '', regex=True)
        X['ScreenType'] = X['ScreenType'].replace(np.nan, X['ScreenType'].mode()[0])

        # Drop ScreenResolution
        X = X.drop('ScreenResolution', axis=1)
        
        # Extract CpuFrequency
        X['CpuFrequency'] = X['Cpu'].str.extract(r'(\d+\.?\d*GHz)').replace('GHz', '', regex=True).astype(float)
        X['Cpu'] = X['Cpu'].str.replace(r'\d+\.?\d*GHz', '', regex=True)
        
        # Convert Ram
        X['Ram'] = X['Ram'].str.replace('GB', '').astype(int)
        
        # Process Memory
        X['Memory'] = X['Memory'].str.replace('1.0TB', '1000GB').str.replace('1TB', '1000GB').str.replace('2TB', '2000GB').str.replace('GB', '')
        X['Memory'] = X['Memory'].str.replace(' ', '')
        
        # Extract storageDisk1 and storageDisk2
        X['storageDisk1'] = X['Memory'].str.extract(r'(^\d+)').astype(int)
        X['storageDisk2'] = X['Memory'].str.extract(r'(\+\d+)')
        X['storageDisk2'] = X['storageDisk2'].fillna('0').str.replace('+', '').astype(int)

        # Extract TypeDisk1 and TypeDisk2
        TypeDisk1 = []
        TypeDisk2 = []
        for i in X['Memory']:
            if len(re.findall(r'\+', i)) == 1:
                allTypes = re.findall(r'([A-z]+)', i)
                TypeDisk1.append(allTypes[0])
                TypeDisk2.append(allTypes[1])
            else:
                allTypes = re.findall(r'([A-z]+)', i)
                TypeDisk1.append(allTypes[0])
                TypeDisk2.append(np.nan)
        
        X['TypeDisk1'] = TypeDisk1
        X['TypeDisk2'] = TypeDisk2
        X['TypeDisk2'] = X['TypeDisk2'].fillna('NaN')
        
        # Drop Memory column
        X = X.drop(columns=['Memory'], axis=1)
        
        # Convert Weight to numeric
        X['Weight'] = X['Weight'].str.replace('kg', '').astype(float)
        
        return X
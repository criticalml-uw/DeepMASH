import pandas as pd
import numpy as np

##import datasets
staticall = pd.read_excel("data/cand_tx_combine.xlsx")
longidata= pd.read_sas('data/stathist_liin.sas7bdat')

##assign nan values for status 998
staticall[staticall==998] = np.nan
staticall[staticall==996] = np.nan

##convert features
def abo(row):
    if row['CAN_ABO'] in ['A','A1','A2']:
        return 'A'
    if row['CAN_ABO'] in ['A1B','A2B','AB']:
        return 'AB'
    if row['CAN_ABO'] =='B':
        return 'B'
    else:
        return 'O'     
staticall["CAN_ABO"] = staticall.apply (lambda row: abo(row), axis=1)


def malig_ty(row):
    MALIG_TY=[1,2,4,8,16,32,64,128,256,512,2048,4096,8192]
    
    if row['CAN_MALIG_TY'] in MALIG_TY:
        return row['CAN_MALIG_TY']
    else:
        return 1024
staticall["CAN_MALIG_TY"] = staticall.apply (lambda row: malig_ty(row), axis=1)


def combine_work_empl(row):
    if (row['CAN_EMPL_STAT']==1) &(row['CAN_WORK_YES_STAT']==1):
        return 1
    if (row['CAN_EMPL_STAT'] in [2,4])|(row['CAN_WORK_YES_STAT'] in [4,5,6,7]):
        return 2
    if (row['CAN_EMPL_STAT'] in [3,6])|(row['CAN_WORK_YES_STAT'] in [2,3])|(row['CAN_WORK_NO_STAT'] in [2,996]):
        return 3
    if (row['CAN_EMPL_STAT'] in [5,7,8,9,996]) |(row['CAN_WORK_NO_STAT'] in [1,3,4,5,6,7,8]):
        return 4
    else:
        return row['CAN_EMPL_STAT'] 
staticall['CAN_EMPL_STAT'] = staticall.apply (lambda row:combine_work_empl(row), axis=1)


def functn_stat(row):
    if row['CAN_FUNCTN_STAT'] in [2010,2020,2030,2040,2050,3]:
        return 3
    if row['CAN_FUNCTN_STAT'] in [2060,2070,2080,2]:
        return 2
    if row['CAN_FUNCTN_STAT'] in [2090,2100,1]:
        return 1
    else:
        return row['CAN_FUNCTN_STAT']
staticall['CAN_FUNCTN_STAT'] = staticall.apply (lambda row: functn_stat(row), axis=1)


def peptic_ulcer(row):
    if row['CAN_PEPTIC_ULCER'] in [2,3,4]:
        return 1
    if row['CAN_PEPTIC_ULCER'] ==1:
        return 0
    else:
        return 0.5
staticall["CAN_PEPTIC_ULCER"] = staticall.apply (lambda row: peptic_ulcer(row), axis=1)

def secondary_pay(row):
    if row['CAN_SECONDARY_PAY'] in [2, 3, 4, 5, 6, 11, 12]:
        return 2
    else:
        return row['CAN_SECONDARY_PAY'] 
        staticall['CAN_SECONDARY_PAY']=staticall.apply(lambda row: secondary_pay(row), axis=1)

def med_cond(row):
    if row['CAN_MED_COND'] in [2,3]:
        return 2
    if row['CAN_MED_COND'] == 1:
        return 1
    else:
        return 0.5
staticall['CAN_MED_COND'] = staticall.apply(lambda row: med_cond(row), axis=1)

def diab(row):
    if row['CAN_DIAB'] in [2,3,4]:
        return 1
    if row['CAN_DIAB'] == 1:
        return 0
    else:
        return 0.5
staticall['CAN_DIAB'] = staticall.apply(lambda row: diab(row), axis=1)


##Feautre processor
class FeatureExtractor():
    
    def extract_total_features(self, candata):
        """ 
        Accepts as input a dataframe of the form in candata. Drops columns
        containing information that is unavailable to physicians at the time of
        listing, and formats columns appropriately so that a learning model
        may be trained on them.
        """
        processor = FeatureProcessor()

        out = pd.DataFrame(candata["PX_ID"])
        # Then we append columns as needed
        
        #age
        out = pd.concat([out, candata["CAN_AGE_IN_MONTHS_AT_LISTING"]], axis=1)
        
        #gender
        out = pd.concat([out,
            processor.gender_binary(candata["CAN_GENDER"])], axis=1)
    
        #blood type
        out = pd.concat([out, 
            pd.get_dummies(candata["CAN_ABO"], prefix="CAN_ABO")], axis=1)
        
        
        #medical/physical condition
        #continous, fill na with mean
        out = pd.concat([out, candata["CAN_WGT_KG"].fillna(candata["CAN_WGT_KG"].mean())], axis=1)
        out = pd.concat([out,candata["CAN_BMI"].fillna(candata["CAN_BMI"].mean())], axis=1)
        out = pd.concat([out, candata["CAN_INIT_ACT_STAT_CD"].fillna(candata["CAN_INIT_ACT_STAT_CD"].mean())], axis=1)
        out = pd.concat([out, candata["CAN_TOT_ALBUMIN"].fillna(candata["CAN_TOT_ALBUMIN"].mean())], axis=1)
        
        
        #categorical 
        out = pd.concat([out, 
            pd.get_dummies(candata["CAN_FUNCTN_STAT"], prefix="CAN_FUNCTN_STAT")], axis=1)
        out = pd.concat([out, candata["CAN_MED_COND"]], axis=1)
        out = pd.concat([out, 
            processor.yesnounknown_to_numeric(candata["CAN_MALIG"])], axis=1)
        out = pd.concat([out, 
            pd.get_dummies(candata["CAN_MALIG_TY"], prefix="CAN_MALIG_TY")], axis=1)
        out = pd.concat([out, 
            processor.yesno_to_numeric(candata["CAN_LIFE_SUPPORT"])],axis=1)
        out = pd.concat([out, candata["CAN_LIFE_SUPPORT_OTHER"]], axis=1)
        out = pd.concat([out, 
            pd.get_dummies(candata["CAN_PHYSC_CAPACITY"], prefix="CAN_PHYSC_CAPACITY")], axis=1)
        out = pd.concat([out, 
            processor.yesnounknown_to_numeric(candata["CAN_PREV_TXFUS"])], axis=1)
        out = pd.concat([out, 
            processor.yesnounknown_to_numeric(candata["CAN_PREV_ABDOM_SURG"])], axis=1)
        out = pd.concat([out, candata["CAN_VENTILATOR"]], axis=1)
        
        
        #comorbidities
        out = pd.concat([out,
            pd.get_dummies(candata["CAN_ANGINA"], prefix="CAN_ANGINA")],axis=1)
        out = pd.concat([out,
            pd.get_dummies(candata["CAN_ANGINA_CAD"], prefix="CAN_ANGINA_CAD")],axis=1)
        out = pd.concat([out, 
            processor.yesnounknown_to_numeric(candata["CAN_ASCITES"])], axis=1)
        out = pd.concat([out, 
            processor.yesnounknown_to_numeric(candata["CAN_BACTERIA_PERIT"])], axis=1)
        out = pd.concat([out, 
            processor.yesnounknown_to_numeric(candata["CAN_CEREB_VASC"])], axis=1)
        out = pd.concat([out,
            pd.get_dummies(candata["CAN_DIAB_TY"], prefix="CAN_DIAB_TY")],axis=1)
        out = pd.concat([out, candata["CAN_DIAB"]], axis=1)
        out = pd.concat([out,
            pd.get_dummies(candata["CAN_DIAL"], prefix="CAN_DIAL")],axis=1)
        out = pd.concat([out, 
            processor.yesnounknown_to_numeric(candata["CAN_DRUG_TREAT_HYPERTEN"])], axis=1)
        out = pd.concat([out, 
            processor.yesnounknown_to_numeric(candata["CAN_DRUG_TREAT_COPD"])], axis=1)
        out = pd.concat([out, 
            processor.yesnounknown_to_numeric(candata["CAN_ENCEPH"])], axis=1)
        out = pd.concat([out, 
            processor.yesnounknown_to_numeric(candata["CAN_MUSCLE_WASTING"])], axis=1)
        out = pd.concat([out, candata["CAN_PEPTIC_ULCER"]], axis=1)
        out = pd.concat([out, 
            processor.yesnounknown_to_numeric(candata["CAN_PERIPH_VASC"])], axis=1)
        out = pd.concat([out, 
            processor.yesnounknown_to_numeric(candata["CAN_PORTAL_VEIN"])], axis=1)
        out = pd.concat([out, 
            processor.yesnounknown_to_numeric(candata["CAN_PULM_EMBOL"])], axis=1)
        out = pd.concat([out, 
            processor.yesnounknown_to_numeric(candata["CAN_TIPSS"])], axis=1)
        out = pd.concat([out, 
            processor.yesnounknown_to_numeric(candata["CAN_VARICEAL_BLEEDING"])], axis=1)
        
        return out
        
class DynamicFeatureExtractor():
    
    def extract_features(self, df):
        df = df[~df["CAN_INIT_SRTR_LAB_MELD"].isnull()]

        processor = FeatureProcessor()

        out = pd.DataFrame(df["PX_ID"])
        # Add in the reference features first
        out = pd.concat([out, df["CANHX_BEGIN_DT"]], axis=1)
        out = pd.concat([out, df["CAN_LISTING_DT"]], axis=1)

        # Add in the covariate features for MELD calculations
        out = pd.concat([out, df["CANHX_ALBUMIN"].fillna(df["CANHX_ALBUMIN"].mean())], axis=1)
        out = pd.concat([out, df["CANHX_BILI"].fillna(df["CANHX_BILI"].mean())], axis=1)
        out = pd.concat([out, df["CANHX_INR"].fillna(df["CANHX_INR"].mean())], axis=1)
        out = pd.concat([out, df["CANHX_SERUM_CREAT"].fillna(df["CANHX_SERUM_CREAT"].mean())], axis=1)
        out = pd.concat([out, df["CANHX_SERUM_SODIUM"].fillna(df["CANHX_SERUM_SODIUM"].mean())], axis=1)
        
        return out

class FeatureProcessor():
    """
    Utility class to gather together feature processing functions.
    """
    def yesno_to_numeric(self, col):
        mapping = {'N': 0, 'Y': 1}
        def safe_mapping(key):
            try:
                return mapping[key]
            except KeyError:
                return -1
        return col.apply(safe_mapping)

    def yesnounknown_to_numeric(self, col):
        mapping = {'N': 0, 'Y': 1, 'U': 0.5}
        def safe_mapping(key):
            try:
                return mapping[key]
            except KeyError:
                return 0.5 # If we don't know - assume that it's average
        return col.apply(safe_mapping)
    
    def yesnounknown_to_numeric_a(self, col):
        mapping = {b'N': 0, b'Y': 1, b'A': 0.5}
        def safe_mapping(key):
            try:
                return mapping[key]
            except KeyError:
                return 0.5 # If we don't know - assume that it's average
        return col.apply(safe_mapping)

    def ordinal_meanpadding(self, col, exceptions):
        def apply_ordinal(elem):
            if elem in exceptions:
                return np.nan
            return elem
        return col.apply(apply_ordinal).fillna(col.mean())

    def gender_binary(self, col):
        return col.apply(lambda x : 1 if x == 'F' else 0)       


##extract static features
extractor = FeatureExtractor()

static_features = extractor.extract_total_features(staticall)

##extract longitudinal features
dynamicextractor = DynamicFeatureExtractor()
longi_features = dynamicextractor.extract_features(longidata)

longi_features['CANHX_BILI']=longi_features['CANHX_BILI']*17.1
longi_features['CANHX_SERUM_CREAT']=longi_features['CANHX_SERUM_CREAT']*88.42
longi_features['CANHX_ALBUMIN']=longi_features['CANHX_ALBUMIN']*10

##combine variables and save
df= pd.merge(static_features, longi_features,on="PX_ID", how = "left")

static_subset= df[['PX_ID','CAN_AGE_AT_LISTING', 'CAN_GENDER', 'CAN_ABO_A',
       'CAN_ABO_AB', 'CAN_ABO_B', 'CAN_ABO_O', 'CAN_WGT_KG', 'CAN_BMI',
       'CAN_INIT_ACT_STAT_CD', 
       'CAN_MED_COND','CAN_FUNCTN_STAT_1.0',
       'CAN_FUNCTN_STAT_2.0', 'CAN_FUNCTN_STAT_3.0',
       'CANHX_ALBUMIN','CANHX_BILI',
       'CANHX_INR','CANHX_SERUM_CREAT',
       'CANHX_SERUM_SODIUM',                   
       'event','wl_to_event']] # to match the list of variables shared with UHN dataset 

static_subset.to_csv('processed_data.csv', index=False)

print('Complete')

import pandas as pd
df = pd.read_excel("../data/raw/GSAF5.xls")
df.head()
df['Injury'] = df['Injury'].fillna('Unknown').astype(str).str.lower().str.strip()


def classify_injury(text):
    text = str(text).lower().strip()

    if any(word in text for word in ['no injury', 'uninjured','unharmed', 'none']):
        return 'No Injury'

    if any(word in text for word in ['survived', 'recovered', 'escaped', 'rescued', 'lived']):
        return 'Severe Wounds'

    if any(word in text for word in [
        'body not recovered', 'fatal', 'died', 'death', 'deceased',
        'human remains', 'presumed dead'
    ]):
        return 'Fatal Wounds'
    
    if any(word in text for word in [
        'minor', 'scratch', 'abrasion', 'bruise', 'small', 'superficial', 'cut'
    ]):
        return 'Minor Wounds'

    if any(word in text for word in [
        'amputation', 'severe', 'major', 'deep', 'critical', 'massive',
        'laceration', 'bite', 'injured', 'tissue loss'
    ]):
        return 'Severe Wounds'

    return 'Unknown'

df['Injury_Class'] = df['Injury'].apply(classify_injury)

def clean_fatal_column(df):
    df['Fatal Y/N'] = (
        df['Fatal Y/N']
        .astype(str)
        .str.strip()
        .str.upper()
        .replace({
            'UNKNOWN': 'N',
            'F': 'N',
            'M': 'N',
            'NAN': 'N',
            'NQ': 'N',
            '2017': 'N',
            'Y X 2': 'Y',
        })
    )
    return df

import pandas as pd

def clean_activity(df):
    df['activity_clean'] = (
        df['Activity']
        .str.lower()
        .str.strip()
        .str.replace(r'[^a-z\s]', '', regex=True)
    )
    return df


def categorize_activity(text):
    if pd.isna(text):
        return "Unknown"
    if any(word in text for word in ["surf", "bodyboard", "paddle", "boogie",
                                     "body boarding", "kiteboarding", "foilboarding",
                                     "skimboarding", "wakeboarding"]):
        return "Surfing"
    elif any(word in text for word in ["swim", "bathing", "snorkel", "rescue",
                                       "float", "splash", "swimming"]):
        return "Swimming"
    elif any(word in text for word in ["fish", "spearfish", "net", "catch", "line",
                                       "fishing", "spear", "scalloping", "lobstering",
                                       "hunt", "clamming"]):
        return "Fishing"
    elif any(word in text for word in ["dive", "scuba", "freediv", "underwater",
                                       "research", "investigat", "pearl", "recover",
                                       "diving"]):
        return "Diving"
    elif any(word in text for word in ["boat", "kayak", "sail", "ship", "vessel",
                                       "frigate", "dinghy", "canoe", "race", "compet",
                                       "rowing", "watercraft", "jet ski", "paddling",
                                       "sculling", "raft", "yacht"]):
        return "Boating"
    elif any(word in text for word in ["walk", "stand", "wade", "reef", "shore",
                                       "beach", "adrift", "wading", "tread"]):
        return "Wading"
    elif any(word in text for word in ["sea disaster", "aircraft", "boeing", "wreck",
                                       "hurricane", "tsunami", "earthquake", "disaster",
                                       "plunged", "sank", "destroyed", "overboard",
                                       "suicide", "air", "petting", "capsize", "swept",
                                       "help", "ride", "sunk", "went down", "crash",
                                       "sinking"]):
        return "Catastrophe"
    else:
        return "Other Activity"


def assign_activity_group(df):
    if 'activity_clean' not in df.columns:
        raise KeyError("DataFrame must have 'activity_clean' column before assigning activity groups.")
    df['Activity_group'] = df['activity_clean'].apply(categorize_activity)
    return df


df['Date'] = df['Date'].astype(str)

import numpy as np

def extract_month_or_nan(value):
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    value_str = str(value)
    for month in months:
        if month in value_str:
            return month
    return np.nan  #assign NaN if no month found


def get_season(month):
    if month in ["Dec", "Jan", "Feb"]:
        return "Winter"
    elif month in ["Mar", "Apr", "May"]:
        return "Spring"
    elif month in ["Jun", "Jul", "Aug"]:
        return "Summer"
    elif month in ["Sep", "Oct", "Nov"]:
        return "Autumn"
    else:
        return np.nan


def assign_season(df):
    df['Date'] = df['Date'].astype(str)
    df['Extracted_Month_NaN'] = df['Date'].apply(extract_month_or_nan)
    df['Season'] = df['Extracted_Month_NaN'].apply(get_season)
    return df

import re

def clean_location_and_state(df):
    # Fill missing values first
    df['Location'] = df['Location'].fillna('Unknown')
    df['State'] = df['State'].replace('', 'Unknown')

    # Define unwanted words and compile regex
    unwanted_words = [
        'miles', 'mi', 'km', 'kilometers', 'islands', 'near', 'nm', 'off', 
        'below', 'the', 'of', 'east of', 'west of', 'south of', 'between', 
        '&', 'north of', 'ºN', 'ºS', 'ºW', 'ºE'
    ]
    pattern = r'\b(?:' + '|'.join(unwanted_words) + r'|\d+)\b'

    # Clean Location
    df['location_clean'] = (
        df['Location']
        .str.replace(pattern, '', regex=True, flags=re.IGNORECASE)
        .str.replace(r'\s+', ' ', regex=True)
        .str.strip()
    )
    df['location_clean'] = df['location_clean'].replace('', 'Unknown').fillna('Unknown')

    # Clean State
    df['state_clean'] = (
        df['State']
        .str.replace(pattern, '', regex=True, flags=re.IGNORECASE)
        .str.replace(r'\s+', ' ', regex=True)
        .str.strip()
    )
    df['state_clean'] = df['state_clean'].replace('', 'Unknown').fillna('Unknown')

    return df


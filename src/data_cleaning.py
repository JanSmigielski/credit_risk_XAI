import numpy as np 
import pandas as pd 
import re
from pathlib import Path

RAW_DIR = Path("data/raw")
CLEAN_DIR = Path("data/cleaned")
# Reading the data from financial statements 
data_2022 = pd.read_csv(RAW_DIR / "data_2022.csv")
data_2023 = pd.read_csv(RAW_DIR / "data_2023.csv")
altman_2023 = data_2023[['KRS', 'Wskaźnik Altmana']].rename(columns={'Wskaźnik Altmana': 'Wskaźnik Altmana 2023'})
data_2022_def = pd.merge(data_2022, altman_2023, on="KRS", how="left")
# Reading the datasets with bankrupcy and restructurization data 
data_bankrupcy = pd.read_excel(RAW_DIR /'default.xlsx', sheet_name = 'upadlosci')
data_restructurization = pd.read_excel(RAW_DIR /'default.xlsx', sheet_name = 'restrukt')


# Creating tables with default information from bankrupcy data and dropping all observations with missing ID 
bankrupcy_KRS = data_bankrupcy[['KRS']]
bankrupcy_KRS['KRS_def'] = 1
bankrupcy_KRS = bankrupcy_KRS.dropna()
bankrupcy_KRS = bankrupcy_KRS.drop_duplicates()

bankrupcy_REGON = data_bankrupcy[['Regon']].rename(columns = {'Regon': 'REGON'})
bankrupcy_REGON['REGON_def'] = 1
bankrupcy_REGON = bankrupcy_REGON.dropna()
bankrupcy_REGON = bankrupcy_REGON.drop_duplicates()

bankrupcy_NIP = data_bankrupcy[['NIP']]
bankrupcy_NIP['NIP_def'] = 1
bankrupcy_NIP = bankrupcy_NIP.dropna()
bankrupcy_NIP = bankrupcy_NIP.drop_duplicates()

# Creating tables with default information from restructurization data and dropping all observations with missing ID 
restr_KRS = data_restructurization[['KRS']]
restr_KRS['KRS_restr'] = 1
restr_KRS = restr_KRS.dropna()
restr_KRS = restr_KRS.drop_duplicates()

restr_REGON = data_restructurization[['Regon']].rename(columns = {'Regon': 'REGON'})
restr_REGON['REGON_restr'] = 1
restr_REGON = restr_REGON.dropna()
restr_REGON = restr_REGON.drop_duplicates() 

restr_NIP = data_restructurization[['NIP']]
restr_NIP['NIP_restr'] = 1
restr_NIP = restr_NIP.dropna() 
restr_NIP = restr_NIP.drop_duplicates()

# Joining all default sources to the main table 
data_2022_def = pd.merge(data_2022_def, bankrupcy_KRS, on='KRS', how='left')
data_2022_def = pd.merge(data_2022_def, bankrupcy_REGON, on='REGON', how='left')
data_2022_def = pd.merge(data_2022_def, bankrupcy_NIP, on='NIP', how='left')
data_2022_def = pd.merge(data_2022_def, restr_KRS, on='KRS', how='left')
data_2022_def = pd.merge(data_2022_def, restr_REGON, on='REGON', how='left')
data_2022_def = pd.merge(data_2022_def, restr_NIP, on='NIP', how='left')

# Assigning default qualitative flag basing on bankrupcy data 
data_2022_def['def_qual'] = data_2022_def[['KRS_def', 'REGON_def', 'NIP_def', 'KRS_restr', 'REGON_restr', 'NIP_restr']].notna().any(axis=1).astype(int)

# Dropping observations with no default information on reporting date 
data_2022_def = data_2022_def[data_2022_def['Wskaźnik Altmana'].notnull()]

# Filling missing values for 2023 default data with some high value to ensure they're not treated as a default 
data_2022_def['Wskaźnik Altmana 2023'].fillna(999999, inplace=True)

# Sorting Altman values ascending 
sorted_values = data_2022_def[['Wskaźnik Altmana', 'Wskaźnik Altmana 2023']].stack().sort_values().values

# Finding default threshold 
threshold = None
for val in sorted_values:
    data_2022_def['default'] = (data_2022_def['Wskaźnik Altmana'] <= val).astype(int)
    data_2022_def['default_2023'] = (data_2022_def['Wskaźnik Altmana 2023'] <= val).astype(int)
    
    count_default_0 = (data_2022_def['default'] == 0).sum()
    count_default_2023_1 = len(data_2022_def[(data_2022_def['default_2023'] == 1) & (data_2022_def['default'] == 0)])
    
    if count_default_0 > 0 and (abs((count_default_2023_1 / count_default_0) - 0.065) < 0.001):
        threshold = val
        break

# Assignign final default flag 
if threshold is not None:
    data_2022_def['default'] = (data_2022_def['Wskaźnik Altmana'] <= threshold).astype(int)
    data_2022_def['default_2023'] = (data_2022_def['Wskaźnik Altmana 2023'] <= threshold).astype(int)
    print(f"Ustalony próg: {threshold}")
else:
    print("Nie znaleziono odpowiedniego progu.")
    
# Creating final into default flag basing on Altman score and bankrupcy data 
data_2022_def["into_default"] = (data_2022_def["default"] | data_2022_def["default_2023"]).astype(int)

# Creating final dataset with only performing entities for year 2022 
data_final = data_2022_def[data_2022_def['default'] == 0]

# Calculating percentage of missing values for each column 
missing_percentages = (data_final.isnull().sum() / len(data_final)) * 100

# Creating an object with columns to keep in modelling dataset (maximum 50% missings)
columns_to_keep = missing_percentages[missing_percentages <= 50].index

# Creating dataset with no rejected columns 
data_final_cleaned = data_final[columns_to_keep]

# Creating a list of irrelevant columns 
irrelevant_columns = ['Unnamed: 0', 'Num', 'Kraj', 'Firma', 
                      'Beneficjent rzeczywisty - Nazwa', 
                      'Beneficjent rzeczywisty - Inne uprawnienia', 
                      'Miasto', 'Kod pocztowy', 'Adres', 'Telefon', 
                      'Fax', 'E-mail', 'Strona www', 'Typ adresu', 
                      'Władze firmy', 'Status działalności', 
                      'Udziałowcy', 'Wskaźnik Altmana', 'Rok obrotowy',
                      'Źródło', 'EMIS ID', 'ISIN', 'NIP', 'REGON', 
                      'Ticker Symbol', 'Wskaźnik Altmana 2023', 
                      'def_qual', 'default_2023']

data_final_cleaned = data_final_cleaned.drop(columns=irrelevant_columns)

# Mapping industries 
category_mapping = {
    "nieruchomości": ["nieruchomości", "zarządzanie nieruchomościami", "budownictwo"],
    "handel": ["sprzedaż hurtowa", "sprzedaż detaliczna", "supermarkety", "domy towarowe"],
    "transport": ["transport", "logistyka"],
    "finanse": ["rachunkowość", "doradztwo podatkowe", "zarządzanie finansami"],
    "opieka zdrowotna": ["szpitale", "opieka zdrowotna", "usługi medyczne"],
    "usługi techniczne": ["inżynieryjne", "projektowanie systemów", "informatyka"],
    "produkcja": ["produkcja", "fabryka", "przemysł"],
    "rolnictwo": ["uprawa", "rolnictwo", "gospodarka leśna"],
    "marketing": ["reklama", "agencje reklamowe", "marketing"],
    "usługi naprawcze": ["naprawa", "konserwacja", "serwis"],
}

def map_category(activity):
    if pd.isna(activity):
        return "inne"
    
    activity_lower = str(activity).lower()
    for key, values in category_mapping.items():
        if any(val in activity_lower for val in values):
            return key
    return "inne" 

data_final_cleaned["Industry"] = data_final_cleaned["Główne obszary działalności (NAICS)"].apply(map_category)

# Correction of one observation's open date - from 2017 to 2017-06-30
data_final_cleaned.loc[data_final_cleaned["Data założenia/wpisu do rejestru"] == "2017", "Data założenia/wpisu do rejestru"] = "2017-06-30"

data_final_cleaned["Data założenia/wpisu do rejestru"] = pd.to_datetime(data_final_cleaned["Data założenia/wpisu do rejestru"])

data_final_cleaned["Wiek firmy (lata)"] = (pd.to_datetime("2022-12-31") - data_final_cleaned["Data założenia/wpisu do rejestru"]).dt.days // 365

data_final_cleaned["Wiek firmy (lata)"].fillna(0, inplace=True)

irrelevant_columns_2 = ['Sektory (NAICS)', 'Główne obszary działalności (NAICS)', 'Pozostała działalność (NAICS)', 
                        'Sektory (EMIS Industries)', 'Główne obszary działalności (EMIS Industries)', 'Pozostała działalność (EMIS Industries)', 
                        'Sektory (PKD 2007)', 'Główne obszary działalności (PKD 2007)', 'Pozostała działalność (PKD 2007)', 'Import', 
                        'Eksport', 'Data założenia/wpisu do rejestru']
data_final_cleaned = data_final_cleaned.drop(columns=irrelevant_columns_2)


# Lista 16 polskich województw
wojewodztwa = [
    "dolnośląskie", "kujawsko-pomorskie", "lubelskie", "lubuskie", "łódzkie", 
    "małopolskie", "mazowieckie", "opolskie", "podkarpackie", "podlaskie", 
    "pomorskie", "śląskie", "świętokrzyskie", "warmińsko-mazurskie", 
    "wielkopolskie", "zachodniopomorskie"
]

# Normalizacja nazw (małe litery, usunięcie zbędnych spacji)
data_final_cleaned["Stan/Województwo"] = data_final_cleaned["Stan/Województwo"].astype(str).str.strip().str.lower()

# Zamiana wartości spoza listy województw oraz pustych na "NA"
data_final_cleaned["Stan/Województwo"] = data_final_cleaned["Stan/Województwo"].apply(lambda x: x if x in wojewodztwa else "Nieznane")

# Zamiana pustych wartości w kolumnie "Udziały skarbu państwa" na 0
data_final_cleaned["Udziały skarbu państwa"] = data_final_cleaned["Udziały skarbu państwa"].fillna(0)

data_final_cleaned.loc[(data_final_cleaned["Dług"].isnull()) & (data_final_cleaned["Dług netto"] == 0), "Dług"] = 0

# Deleting columns as they cannot be calculated when debt = 0
to_delete = ['Wskaźnik pokrycia zobowiązań (x)', 'Przepływy pieniężne netto/Dług (%)']
data_final_cleaned = data_final_cleaned.drop(columns=to_delete)

# Dropping columns as they cannot be calculated without additional information 
to_delete = ['Dynamika zapasów (%)', 'Dynamika rzeczowych aktywów trwałych netto (%)', 'Dynamika EBITDA (%)', 
             'Dynamika wartości księgowej (%)', 'Dynamika należności (%)', 'Dynamika przychodów ze sprzedaży netto (%)', 
             'Dynamika wyniku operacyjnego (%)', 'Dynamika przychodów ogółem (%)', 'Dynamika zysku netto (%)', 
             'Dynamika kapitałów własnych (%)', 'Dynamika aktywów ogółem (%)'] 
data_final_cleaned = data_final_cleaned.drop(columns=to_delete)

to_delete = ['Różnice kursowe z przeliczenia jednostek podporządkowanych', 'Rezerwa z tytułu aktualizacji wyceny', 
             'Podatek dochodowy/Sprzedaż netto (%)', 'Wskaźnik udziału eksportu (%) (%)', 
             'Wynagrodzenia i świadczenia na rzecz pracowników/Sprzedaż netto (%)', 
             'Przepływy środków pieniężnych z działalności inwestycyjnej', 
             'Przepływy środków pieniężnych z działalności finansowej', 'Wartości niematerialne i prawne oraz wartość firmy', 
             'Rzeczowe aktywa trwałe']
data_final_cleaned = data_final_cleaned.drop(columns=to_delete)

to_delete = ['Koszty administracyjne/Sprzedaż netto (%)', 'Wskaźnik rotacji zapasów (x)', 'Wskaźnik rotacji aktywów trwałych (x)', 
             'Wzrost (spadek) netto środków pieniężnych', 'Środki pieniężne na koniec okresu', 'Środki pieniężne na początek okresu', 
             'Kapitały mniejszości']
data_final_cleaned = data_final_cleaned.drop(columns=to_delete)

# Deleted as cannot be calculated 
to_delete = ['Wskaźnik pokrycia odsetek (x)']
data_final_cleaned = data_final_cleaned.drop(columns=to_delete)

to_delete = ['EBITDA/Przychody netto ze sprzedaży (%)', 'Wskaźnik rotacji zobowiązań (x)', 'Wskaźnik szybkiej płynności (x)', 
             'Wskaźnik przepływów operacyjnych (x)', 'Wskaźnik płynności gotówkowej (x).1', 'Wskaźnik płynności gotówkowej (x)', 
             'Wskaźnik bieżącej płynności (x)', 'Wynik finansowy (zysk / strata)', 'Przepływy środków pieniężnych z działalności operacyjnej', 
             'Zyski zatrzymane', 'Wskaźnik obrotu należnościami (x)', 'Marża z wyniku operacyjnego (%)', 'Rentowność netto (%)', 
             'Marża zysku brutto ze sprzedaży (%)', 'Przepływy pieniężne z działalności operacyjnej/Przychody ze sprzedaży (%)', 
             'Deprecjacja i amortyzacja/Sprzedaż netto (%)', 'Odsetki zapłacone/ Sprzedaż netto (%)']
data_final_cleaned = data_final_cleaned.drop(columns=to_delete)

# Debt/EBITDA is null when EBIT = 0, minimum value will be used as an approximation 
data_final_cleaned['Dług/EBITDA (x)'] = data_final_cleaned['Dług/EBITDA (x)'].fillna(min(data_final_cleaned['Dług/EBITDA (x)']))

# Deleted as cannot be calculated when assets = 0
to_delete = ['Przepływy pieniężne z działalności operacyjnej/Kapitał własny (%)']
data_final_cleaned = data_final_cleaned.drop(columns=to_delete)

# Deleted as cannot be calculated when EBIT = 0
to_delete = ['Przepływy pieniężne z działalności operacyjnej/EBIT (%)']
data_final_cleaned = data_final_cleaned.drop(columns=to_delete)

data_final_cleaned_1 = data_final_cleaned.dropna()

data_final_cleaned = data_final_cleaned_1.copy()

def clean_employment_value_fixed(value):
    """Cleans and converts employment values to numeric, handling comma separators."""
    # Remove year information (values in parentheses)
    value = re.sub(r'\(\d{4}\)', '', value).strip()

    # Replace comma separators in numbers
    value = value.replace(',', '')

    # Handle ranges (e.g., "101 - 250")
    if '-' in value:
        low, high = map(int, value.split('-'))
        return (low + high) / 2  # Calculate the mean of the range

    # Handle "Powyżej ..." cases (e.g., "Powyżej 250")
    if "Powyżej" in value:
        return int(value.split()[-1])  # Take the number after "Powyżej"

    # Convert single numbers to integers
    return int(value)

# Apply transformation again with the fixed function
data_final_cleaned['Liczba zatrudnionych'] = data_final_cleaned['Liczba zatrudnionych'].apply(clean_employment_value_fixed)

# Verify the transformation
output_file = CLEAN_DIR / "data_cleaned.csv"
data_final_cleaned.dropna().to_csv(output_file)

import pandas as pd
from tqdm import tqdm
import math


# where I/O files are located
base_path = 'data'

# Load the data from this excel file
df = pd.read_excel(f'{base_path}/LDM_metadata.xlsx')


def get_manufacturer(manufacturer):
    if str(manufacturer) == '0':
        return 'GE Medical Systems'
    elif str(manufacturer) == '1':
        return 'MPTronic software'
    elif str(manufacturer) == '2':
        return 'Siemens'
    else:
        return ''

def get_scanner(scanner):
    if str(scanner) == '0':
        return 'Avanto'
    elif str(scanner) == '1':
        return "Optima MR4502"
    elif str(scanner) == '2':
        return 'SIGNA EXCITE'
    elif str(scanner) == '3':
        return 'SIGNA HDx'
    elif str(scanner) == '4':
        return 'SIGNA HDxt'
    elif str(scanner) == '5':
        return 'Skyra'
    elif str(scanner) == '6':
        return 'Trio'
    elif str(scanner) == '7':
        return 'TrioTim'
    else:
        return ''

def get_field_strength(field_strength):
    if str(field_strength) == '0':
        return '1.494T'
    elif str(field_strength) == '1':
        return '1.5T'
    elif str(field_strength) == '2':
        return '2.89T'
    elif str(field_strength) == '3':
        return '3T'
    else:
        return ''

# Contrast agents: GADAVIST=0,MAGNEVIST=1,MMAGNEVIST=2,MULTIHANCE=3,Name of agent not stated(but ContrastBolusAgent tag was present)=4, ContrastBolusAgent Tag Absent = 5
def get_contrast_agent(contrast_agent):
    if str(contrast_agent) == '0':
        return 'GADAVIST'
    elif str(contrast_agent) == '1':
        return 'MAGNEVIST'
    elif str(contrast_agent) == '2':
        return 'MMAGNEVIST'
    elif str(contrast_agent) == '3':
        return 'MULTIHANCE'
    elif str(contrast_agent) == '4':
        #return 'Name of agent not stated(but ContrastBolusAgent tag was present)'
        return ''
    elif str(contrast_agent) == '5':
        #return 'ContrastBolusAgent Tag Absent'
        return ''
    else:
        return ''

# Contrast Bolus volumes: 6=0,7=1,8=2,9=3,10=4,11=5,11.88=6,12=7,13=8,13.6=9,14=10,14.5=11,15=12,16=13,17=14,18=15,19=16,20=17,25=18
def get_contrast_bolus_volume(contrast_bolus_volume):
    if str(contrast_bolus_volume) == '0':
        return '6'
    elif str(contrast_bolus_volume) == '1':
        return '7'
    elif str(contrast_bolus_volume) == '2':
        return '8'
    elif str(contrast_bolus_volume) == '3':
        return '9'
    elif str(contrast_bolus_volume) == '4':
        return '10'
    elif str(contrast_bolus_volume) == '5':
        return '11'
    elif str(contrast_bolus_volume) == '6':
        return '11.88'
    elif str(contrast_bolus_volume) == '7':
        return '12'
    elif str(contrast_bolus_volume) == '8':
        return '13'
    elif str(contrast_bolus_volume) == '9':
        return '13.6'
    elif str(contrast_bolus_volume) == '10':
        return '14'
    elif str(contrast_bolus_volume) == '11':
        return '14.5'
    elif str(contrast_bolus_volume) == '12':
        return '15'
    elif str(contrast_bolus_volume) == '13':
        return '16'
    elif str(contrast_bolus_volume) == '14':
        return '17'
    elif str(contrast_bolus_volume) == '15':
        return '18'
    elif str(contrast_bolus_volume) == '16':
        return '19'
    elif str(contrast_bolus_volume) == '17':
        return '20'
    elif str(contrast_bolus_volume) == '18':
        return '25'
    else:
        return ''


# (Taking date of diagnosis as day 0) [Functional Check : numeric entries will be negative only, non-numeric ones will be NA or NC ]
def get_age(age):
    try:
        return math.floor((-1 * (int(age)) / 365.2422))
    except:
        return ''

# ethnicities: "{0 = N/A 1 = white, 2 = black, 3 = asian, 4 = native, 5 = hispanic, 6 = multi, 7 = hawa, 8 = amer indian}"
def get_ethnicity(ethnicity):
    if str(ethnicity) == '0':
        return ''
    elif str(ethnicity) == '1':
        return 'white'
    elif str(ethnicity) == '2':
        return 'black'
    elif str(ethnicity) == '3':
        return 'asian'
    elif str(ethnicity) == '4':
        return 'native american'
    elif str(ethnicity) == '5':
        return 'hispanic'
    elif str(ethnicity) == '6':
        return 'multi'
    elif str(ethnicity) == '7':
        return 'hawa'
    elif str(ethnicity) == '8':
        return 'indian american'
    else:
        return ''

# tumor subtype {0 = luminal-like, 1 = ER/PR pos, HER2 pos, 2 = her2, 3 = trip neg}
def get_subtype(subtype):
    if str(subtype) == '0':
        return 'luminal-like'
    elif str(subtype) == '1':
        return 'ER/PR positive, HER2 positive'
    elif str(subtype) == '2':
        return 'her2 positive'
    elif str(subtype) == '3':
        return 'triple negative'
    else:
        return ''

# histologic types: 0=DCIS 1=ductal 2=lobular 3=metaplastic 4=LCIS 5=tubular 6=mixed 7=micropapillary 8=colloid 9=mucinous 10=medullary
def get_tumor_histologic_type(tumor_histologic_type):
    if str(tumor_histologic_type) == '0':
        return 'DCIS'
    elif str(tumor_histologic_type) == '1':
        return 'ductal'
    elif str(tumor_histologic_type) == '2':
        return 'lobular'
    elif str(tumor_histologic_type) == '3':
        return 'metaplastic'
    elif str(tumor_histologic_type) == '4':
        return 'LCIS'
    elif str(tumor_histologic_type) == '5':
        return 'tubular'
    elif str(tumor_histologic_type) == '6':
        return 'mixed'
    elif str(tumor_histologic_type) == '7':
        return 'micropapillary'
    elif str(tumor_histologic_type) == '8':
        return 'colloid'
    elif str(tumor_histologic_type) == '9':
        return 'mucinous'
    elif str(tumor_histologic_type) == '10':
        return 'medullary'
    else:
        return ''


def get_metastatic(metastatic):
    if str(metastatic) == '0':
        return 'non-metastatic'
    elif str(metastatic) == '1':
        return 'metastatic'
    else:
        return ''

 # Side of cancer L=left R=right
def get_tumor_location(tumor_location):
    if str(tumor_location) == 'L':
        return 'left'
    elif str(tumor_location) == 'R':
        return 'right'
    else:
        return ''

# Bilateral breast cancer? { 0=no 1=yes}
def get_tumor_bilateral(tumor_bilateral):
    if str(tumor_bilateral) == '0':
        return 'unilateral'
    elif str(tumor_bilateral) == '1':
        return 'bilateral'
    else:
        return ''


# create a list to store the data after processing in csv format
new_data = []

# iterate over rows in dataframe df
for index, patient_row in tqdm(df.iterrows()):
    patient_dict = {}
    patient_dict['patient_id'] = patient_row[0]
    # DCE-MRI imaging information (B-G)
    patient_dict['acquisition_times'] = patient_row[1] # In seconds passed after pre-contrast acqusition
    patient_dict['manufacturer'] = get_manufacturer(patient_row[2])
    patient_dict['scanner'] = get_scanner(patient_row[3])
    patient_dict['field_strength'] = get_field_strength(patient_row[4])
    patient_dict['contrast_agent'] = get_contrast_agent(patient_row[5])
    patient_dict['contrast_bolus_volume'] = get_contrast_bolus_volume(patient_row[6])
    patient_dict['metastatic'] = get_metastatic(patient_row[9])

    # Tumor Subtype Information (K-Q, J)
    patient_dict['tumor_subtype'] = get_subtype(patient_row[13])
    patient_dict['tumor_histologic_type'] = get_tumor_histologic_type(patient_row[14])
    patient_dict['tumor_location'] = get_tumor_location(patient_row[15])
    patient_dict['tumor_bilateral'] = get_tumor_bilateral(patient_row[16])

    # Patient Demographic Information (H-I)
    patient_dict['age'] = get_age(patient_row[7])
    patient_dict['ethnicity'] = get_ethnicity(patient_row[8])

    # Text 1: DCE-MRI imaging information (B-G)
    # Text 2: Text 1 + Tumor Subtype Information (K-Q, J)
    # Text 3: Text 2 + Patient Demographic Information (H, I)
    if index > 0:
        contrast_bolus_volume_text = f" at a bolus volume of {patient_dict['contrast_bolus_volume']} mL" if patient_dict['contrast_bolus_volume']!='' else ""
        patient_dict['text1'] = f"Breast DCE-MRI acquired by a {patient_dict['manufacturer']} {patient_dict['scanner']} {patient_dict['field_strength']}" \
                                f" scanner with contrast agent {patient_dict['contrast_agent']}{contrast_bolus_volume_text}."

        histologic_type_text = f" and its histologic type is {patient_dict['tumor_histologic_type']}" if patient_dict['tumor_histologic_type']!='' else ""
        patient_dict['text2'] = f"{patient_dict['text1']} MRI Scan shows {patient_dict['tumor_bilateral']} tumor in {patient_dict['tumor_location']} breast." \
                                f" This {patient_dict['metastatic']} tumor is of subtype '{patient_dict['tumor_subtype']}'{histologic_type_text}."

        age_text = f" Patient age is {patient_dict['age']}." if patient_dict['age']!='' else ""
        ethnicity_text = f" Patient ethnicity is {patient_dict['ethnicity']}." if patient_dict['ethnicity']!='' else ""
        patient_dict['text3'] = f"{patient_dict['text2']}{age_text}{ethnicity_text}"

    new_data.append(patient_dict)

# store the data in a csv file
df = pd.DataFrame(new_data)
df.to_csv(f'{base_path}/LDM_metadata.csv', index=False)





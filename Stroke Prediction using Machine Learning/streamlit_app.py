import pickle
import pandas as pd
import streamlit as st

st.image('background_image.png')

st.title('Stroke Prediction')
with open('model.pkl', 'rb') as model_file:
    loaded_model = pickle.load(model_file)

gender_ = st.multiselect('What is your gender?', ['Female', 'Male'], key = 0)
age_ = st.slider('Age', min_value = 0, max_value = 100)
hypertension_ = st.multiselect('Do you have hypertension?', ['No', 'Yes'], key = 1)
heart_disease_ = st.multiselect('Any heart disease in the past?', ['No', 'Yes'], key = 2)
married_ = st.multiselect('Are you married?', ['No', 'Yes'], key = 3)

list1 = ['Private', 'Government', 'Self', 'Childern', 'Never worked']
work_type = st.multiselect('Which type of work do you do?', list1, key=4)
work_type_ = [list1.index(option) for option in work_type]

list2 = ['Rural', 'Urban']
residence_type = st.multiselect('What is your residence type?', list2, key = 5)
residence_type_ = [list2.index(option) for option in residence_type]

glucose_level_ = st.number_input('Enter glucose level', key = 6)
bmi_ = st.number_input('Enter bmi', key = 7)

list3 = ['Never smoked', 'I don\'t know', 'Yes, I smoke', 'Used to smoke']
smoking_status = st.multiselect('Do you smoke?', list3, key = 8)
smoking_status_ = [list3.index(option) for option in smoking_status]

def gender(entry):
   if entry == ['Female']:
      return 1
   else:
      return 0
 
def hypertension(entry):
   if entry == ['Yes']:
      return 1
   else:
      return 0
    
def heart_disease(entry):
   if entry == ['Yes']:
      return 1
   else:
      return 0
    
def married(entry):
   if entry == ['Yes']:
      return 1
   else:
      return 0
 
def normalize_value(value, min_, max_):
    if min_ < value < max_:
       normal_entry = (value - min_)/(max_ - min_)
       return normal_entry
       
def glucose_level(entry):
    return normalize_value(entry, 55.120000, 271.740000) 
       
def bmi(entry):
    return normalize_value(entry, 10.300000, 97.600000)
    
input_data = {
    'gender': gender(gender_),
    'age': age_,
    'hypertension': hypertension(hypertension_),
    'heart_disease': heart_disease(heart_disease_),
    'ever_married': married(married_),
    'work_type': work_type_,
    'Residence_type': residence_type_,
    'avg_glucose_level': glucose_level(glucose_level_), 
    'bmi': bmi(bmi_), 
    'smoking_status': smoking_status_
}

input_df = pd.DataFrame(input_data)
st.dataframe(input_df)

pred = loaded_model.predict(input_df)
if pred == 1: st.write('You have high chance of having heart stroke'); st.image('background_image2.png')
else: st.write('You have less chance of having heart stroke'); st.image('background_image1.png')

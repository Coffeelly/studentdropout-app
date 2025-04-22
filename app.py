import streamlit as st
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier

# Load model from .cbm file
model = CatBoostClassifier()
model.load_model("catboost_model.cbm")


st.title("Prediksi Dropout Mahasiswa")
st.write("Masukkan data mahasiswa untuk memprediksi apakah mereka akan graduate atau dropout.")

# Input untuk kategori, dengan opsi spesifik
marital_status_map = {
    "Single": 1, "Married": 2, "Widower": 3, "Divorced": 4,
    "Facto Union": 5, "Legally Separated": 6
}
marital_status = st.selectbox("Marital Status", list(marital_status_map.keys()))
marital_status_value = marital_status_map[marital_status]

application_mode_map = {
    "1st Phase - General Contingent": 1, "Ordinance No. 612/93": 2, "Special Contingent (Azores)": 5,
    "Holders of Other Higher Courses": 7, "Ordinance No. 854-B/99": 10, "International Student (Bachelor)": 15,
    "Special Contingent (Madeira)": 16, "2nd Phase - General Contingent": 17, "3rd Phase - General Contingent": 18,
    "Different Plan": 26, "Other Institution": 27, "Over 23 Years Old": 39, "Transfer": 42, "Change of Course": 43,
    "Tech Specialization Diploma Holders": 44, "Change of Institution/Course": 51, "Short Cycle Diploma Holders": 53,
    "International Institution/Course Change": 57
}
application_mode = st.selectbox("Application Mode", list(application_mode_map.keys()))
application_mode_value = application_mode_map[application_mode]

course_map = {
    "Biofuel Production Technologies": 33, "Animation and Multimedia Design": 171, "Social Service (Evening)": 8014,
    "Agronomy": 9003, "Communication Design": 9070, "Veterinary Nursing": 9085, "Informatics Engineering": 9119,
    "Equinculture": 9130, "Management": 9147, "Social Service": 9238, "Tourism": 9254, "Nursing": 9500,
    "Oral Hygiene": 9556, "Advertising & Marketing Management": 9670, "Journalism & Communication": 9773,
    "Basic Education": 9853, "Management (Evening)": 9991
}
course = st.selectbox("Course", list(course_map.keys()))
course_value = course_map[course]

daytime_evening_map = {"Daytime": 1, "Evening": 0}
daytime_evening = st.selectbox("Daytime/Evening Attendance", list(daytime_evening_map.keys()))
daytime_evening_value = daytime_evening_map[daytime_evening]

previous_qualification_map = {
    "Secondary Education": 1, "Bachelor's Degree": 2, "Degree": 3, "Master's": 4, "Doctorate": 5,
    "Higher Education (Incomplete)": 6, "12th Year - Not Completed": 9, "11th Year - Not Completed": 10,
    "Other 11th Year": 12, "10th Year": 14, "10th Year - Not Completed": 15, "Basic Education (9th-11th Year)": 19,
    "Basic Education (6th-8th Year)": 38, "Technological Specialization Course": 39, "Higher Education - Degree (1st Cycle)": 40,
    "Professional Higher Technical Course": 42, "Higher Education - Master (2nd Cycle)": 43
}
previous_qualification = st.selectbox("Previous Qualification", list(previous_qualification_map.keys()))
previous_qualification_value = previous_qualification_map[previous_qualification]

nacionality_map = {
    "Portuguese": 1, "German": 2, "Spanish": 6, "Italian": 11, "Dutch": 13, "English": 14, "Lithuanian": 17, "Angolan": 21, "Cape Verdean": 22, "Guinean": 24, "Mozambican": 25, "Santomean": 26, "Turkish": 32, "Brazilian": 41, "Romanian": 62, "Moldovan": 100, "Mexican": 101, "Ukrainian": 103, "Russian": 105, "Cuban": 108, "Colombian": 109
}
nacionality = st.selectbox("Nacionality", list(nacionality_map.keys()))
nacionality_value = nacionality_map[nacionality]

mothers_qualification_map = {
    "Secondary Education - 12th Year of Schooling or Eq.": 1, "Higher Education - Bachelor's Degree": 2, "Higher Education - Degree": 3, "Higher Education - Master's": 4, "Higher Education - Doctorate": 5, "Frequency of Higher Education": 6, "12th Year of Schooling - Not Completed": 9, "11th Year of Schooling - Not Completed": 10, "7th Year (Old)": 11, "Other - 11th Year of Schooling": 12, "10th Year of Schooling": 14, "General Commerce Course": 18, "Basic Education 3rd Cycle (9th/10th/11th Year) or Equiv.": 19, "Technical-Professional Course": 22, "7th Year of Schooling": 26, "2nd Cycle of the General High School Course": 27, "9th Year of Schooling - Not Completed": 29, "8th Year of Schooling": 30, "Unknown": 34, "Can't Read or Write": 35, "Can Read Without Having a 4th Year of Schooling": 36, "Basic Education 1st Cycle (4th/5th Year) or Equiv.": 37, "Basic Education 2nd Cycle (6th/7th/8th Year) or Equiv.": 38, "Technological Specialization Course": 39, "Higher Education - Degree (1st Cycle)": 40, "Specialized Higher Studies Course": 41, "Professional Higher Technical Course": 42, "Higher Education - Master (2nd Cycle)": 43, "Higher Education - Doctorate (3rd Cycle)": 44
}
mothers_qualification_text = st.selectbox("Mother's Qualification", list(mothers_qualification_map.keys()))
mothers_qualification_value = mothers_qualification_map[mothers_qualification_text]  # Konversi ke angka

fathers_qualification_map = {
    "Secondary Education - 12th Year of Schooling or Eq.": 1, "Higher Education - Bachelor's Degree": 2, "Higher Education - Degree": 3, "Higher Education - Master's": 4, "Higher Education - Doctorate": 5, "Frequency of Higher Education": 6, "12th Year of Schooling - Not Completed": 9, "11th Year of Schooling - Not Completed": 10, "7th Year (Old)": 11, "Other - 11th Year of Schooling": 12, "2nd Year Complementary High School Course": 13, "10th Year of Schooling": 14, "General Commerce Course": 18, "Basic Education 3rd Cycle (9th/10th/11th Year) or Equiv.": 19, "Complementary High School Course": 20, "Technical-Professional Course": 22, "Complementary High School Course - Not Concluded": 25, "7th Year of Schooling": 26, "2nd Cycle of the General High School Course": 27, "9th Year of Schooling - Not Completed": 29, "8th Year of Schooling": 30, "General Course of Administration and Commerce": 31, "Supplementary Accounting and Administration": 33, "Unknown": 34, "Can't Read or Write": 35, "Can Read Without Having a 4th Year of Schooling": 36, "Basic Education 1st Cycle (4th/5th Year) or Equiv.": 37, "Basic Education 2nd Cycle (6th/7th/8th Year) or Equiv.": 38, "Technological Specialization Course": 39, "Higher Education - Degree (1st Cycle)": 40, "Specialized Higher Studies Course": 41, "Professional Higher Technical Course": 42, "Higher Education - Master (2nd Cycle)": 43, "Higher Education - Doctorate (3rd Cycle)": 44
}
fathers_qualification_text = st.selectbox("Father's Qualification", list(fathers_qualification_map.keys()))
fathers_qualification_value = fathers_qualification_map[fathers_qualification_text]  # Konversi ke angka

mothers_occupation_map = {
    "Student": 0, "Legislative Power & Executive Bodies": 1, "Specialists in Intellectual & Scientific Activities": 2, "Intermediate Level Technicians & Professions": 3, "Administrative Staff": 4, "Personal Services, Security & Sellers": 5, "Farmers & Skilled Workers in Agriculture, Fisheries & Forestry": 6, "Skilled Workers in Industry, Construction & Craftsmen": 7, "Installation & Machine Operators and Assembly Workers": 8, "Unskilled Workers": 9, "Armed Forces Professions": 10, "Other Situation": 90, "Blank (No Data)": 99, "Health Professionals": 122, "Teachers": 123, "ICT Specialists": 125, "Intermediate Level Science & Engineering Technicians": 131, "Intermediate Level Health Technicians": 132, "Intermediate Level Legal, Social, Sports, Cultural Technicians": 134, "Office Workers & Secretaries": 141, "Data, Accounting, Statistical, Financial Services Operators": 143, "Other Administrative Support Staff": 144, "Personal Service Workers": 151, "Sellers": 152, "Personal Care Workers": 153, "Skilled Construction Workers (Except Electricians)": 171, "Printing, Precision Instrument Manufacturing, Jewelers, Artisans": 173, "Food Processing, Woodworking, Clothing & Other Crafts": 175, "Cleaning Workers": 191,"Unskilled Agriculture, Animal Production, Fisheries & Forestry Workers": 192, "Unskilled Extractive Industry, Construction, Manufacturing & Transport Workers": 193,"Meal Preparation Assistants": 194
}
mothers_occupation_text = st.selectbox("Mother's Occupation", list(mothers_occupation_map.keys()))
mothers_occupation_value = mothers_occupation_map[mothers_occupation_text]  # Konversi ke angka

fathers_occupation_map = {
    "Student": 0, "Legislative Power & Executive Bodies, Directors, Managers": 1, "Specialists in Intellectual & Scientific Activities": 2, "Intermediate Level Technicians & Professions": 3, "Administrative Staff": 4, "Personal Services, Security & Sellers": 5, "Farmers & Skilled Workers in Agriculture, Fisheries & Forestry": 6, "Skilled Workers in Industry, Construction & Craftsmen": 7, "Installation & Machine Operators and Assembly Workers": 8, "Unskilled Workers": 9, "Armed Forces Professions": 10, "Other Situation": 90, "Blank (No Data)": 99, "Armed Forces Officers": 101, "Armed Forces Sergeants": 102, "Other Armed Forces Personnel": 103, "Directors of Administrative & Commercial Services": 112, "Hotel, Catering, Trade & Other Services Directors": 114, "Specialists in Physical Sciences, Mathematics & Engineering": 121, "Health Professionals": 122, "Teachers": 123, "Specialists in Finance, Accounting, Administration & Public Relations": 124, "Intermediate Level Science & Engineering Technicians": 131, "Intermediate Level Health Technicians": 132, "Intermediate Level Legal, Social, Sports, Cultural Technicians": 134, "Information & Communication Technology Technicians": 135, "Office Workers & Secretaries": 141, "Data, Accounting, Statistical, Financial Services Operators": 143, "Other Administrative Support Staff": 144, "Personal Service Workers": 151, "Sellers": 152, "Personal Care Workers": 153, "Protection & Security Services Personnel": 154, "Market-Oriented Farmers & Skilled Agricultural & Animal Production Workers": 161, "Farmers, Livestock Keepers, Fishermen, Hunters, Gatherers": 163, "Skilled Construction Workers (Except Electricians)": 171, "Skilled Workers in Metallurgy, Metalworking & Similar": 172, "Skilled Workers in Electricity & Electronics": 174, "Workers in Food Processing, Woodworking, Clothing & Other Industries": 175, "Fixed Plant & Machine Operators": 181, "Assembly Workers": 182, "Vehicle Drivers & Mobile Equipment Operators": 183, "Unskilled Workers in Agriculture, Animal Production, Fisheries & Forestry": 192, "Unskilled Workers in Extractive Industry, Construction, Manufacturing & Transport": 193, "Meal Preparation Assistants": 194, "Street Vendors & Street Service Providers": 195
}
fathers_occupation_text = st.selectbox("Father's Occupation", list(fathers_occupation_map.keys()))
fathers_occupation_value = fathers_occupation_map[fathers_occupation_text]  # Konversi ke angka


displaced_map = {
  "Yes": 1, "No": 0
}
displaced_text = st.selectbox("Displaced", list(displaced_map.keys()))
displaced_value = displaced_map[displaced_text]

educational_special_needs_map = {
  "Yes": 1, "No": 0
}
educational_special_needs_text = st.selectbox("Educational Special Needs", list(educational_special_needs_map.keys()))
educational_special_needs_value = educational_special_needs_map[educational_special_needs_text]

debtor_map = {
  "Yes": 1, "No": 0
}
debtor_text = st.selectbox("debtor", list(debtor_map.keys()))
debtor_value = debtor_map[debtor_text]

tuition_fees_up_to_date_map = {
  "Yes": 1, "No": 0
}
tuition_fees_up_to_date_text = st.selectbox("Tuition Fees Up To Date", list(tuition_fees_up_to_date_map.keys()))
tuition_fees_up_to_date_value = tuition_fees_up_to_date_map[tuition_fees_up_to_date_text]

gender_map = {
  "Male": 1, "Female": 0
}
gender_text = st.selectbox("Gender", list(gender_map.keys()))
gender_value = gender_map[gender_text]

scholarship_holder_map = {
  "Yes": 1, "No": 0
}
scholarship_holder_text = st.selectbox("Scholarship Holder", list(scholarship_holder_map.keys()))
scholarship_holder_value = scholarship_holder_map[scholarship_holder_text]

international_map = {
  "Yes": 1, "No": 0
}
international_text = st.selectbox("International", list(international_map.keys()))
international_value = international_map[international_text]

# Input untuk numerik dengan batasan sesuai karakteristik data
application_order = st.number_input("Application Order", min_value=0, max_value=9, value=0)
previous_qualification_grade = st.number_input("Previous Qualification Grade", min_value=0.0, max_value=200.0, value=100.0)
admission_grade = st.number_input("Admission Grade", min_value=0.0, max_value=200.0, value=100.0)
age_at_enrollment = st.number_input("Age at Enrollment", min_value=16, max_value=100, value=18)

curr_units_1st_credited = st.number_input("Curricular Units 1st Sem Credited", min_value=0, max_value=20, value=10)
curr_units_1st_enrolled = st.number_input("Curricular Units 1st Sem Enrolled", min_value=0, max_value=26, value=10)
curr_units_1st_evaluations = st.number_input("Curricular Units 1st Sem Evaluations", min_value=0, max_value=45, value=10)
curr_units_1st_approved = st.number_input("Curricular Units 1st Sem Approved", min_value=0, max_value=26, value=10)
curr_units_1st_grade = st.number_input("Curricular Units 1st Sem Grade", min_value=0.0, max_value=20.0, value=10.0)
curr_units_1st_without_eval = st.number_input("Curricular Units 1st Sem Without Evaluations", min_value=0, max_value=12, value=10)

curr_units_2nd_credited = st.number_input("Curricular Units 2nd Sem Credited", min_value=0, max_value=20, value=10)
curr_units_2nd_enrolled = st.number_input("Curricular Units 2nd Sem Enrolled", min_value=0, max_value=26, value=10)
curr_units_2nd_evaluations = st.number_input("Curricular Units 2nd Sem Evaluations", min_value=0, max_value=45, value=10)
curr_units_2nd_approved = st.number_input("Curricular Units 2nd Sem Approved", min_value=0, max_value=26, value=10)
curr_units_2nd_grade = st.number_input("Curricular Units 2nd Sem Grade", min_value=0.0, max_value=20.0, value=10.0)
curr_units_2nd_without_eval = st.number_input("Curricular Units 2nd Sem Without Evaluations", min_value=0, max_value=12, value=10)

unemployment_rate = st.number_input("Unemployment Rate", min_value=0.0, max_value=20.0, value=2.0)
inflation_rate = st.number_input("Inflation Rate", min_value=-1.0, max_value=4.0, value=2.0)
gdp = st.number_input("GDP", min_value=-2.0, max_value=4.0, value=2.0)

# Tombol Prediksi
if st.button("Prediksi"):
    # Simpan input ke dalam dictionary
    input_data = pd.DataFrame([{
        "Marital_status": marital_status_value,
        "Application_mode": application_mode_value,
        "Application_order": application_order,
        "Course": course_value,
        "Daytime_evening_attendance": daytime_evening_value,
        "Previous_qualification": previous_qualification_value,
        "Previous_qualification_grade": previous_qualification_grade,
        "Nacionality": nacionality_value,
        "Mothers_qualification": mothers_qualification_value,
        "Fathers_qualification": fathers_qualification_value,
        "Mothers_occupation": mothers_occupation_value,
        "Fathers_occupation": fathers_occupation_value,
        "Admission_grade": admission_grade,
        "Displaced": displaced_value,
        "Educational_special_needs": educational_special_needs_value,
        "Debtor": debtor_value,
        "Tuition_fees_up_to_date": tuition_fees_up_to_date_value,
        "Gender": gender_value,
        "Scholarship_holder": scholarship_holder_value,
        "Age_at_enrollment": age_at_enrollment,
        "International": international_value,
        "Curricular_units_1st_sem_credited": curr_units_1st_credited,
        "Curricular_units_1st_sem_enrolled": curr_units_1st_enrolled,
        "Curricular_units_1st_sem_evaluations": curr_units_1st_evaluations,
        "Curricular_units_1st_sem_approved": curr_units_1st_approved,
        "Curricular_units_1st_sem_grade": curr_units_1st_grade,
        "Curricular_units_1st_sem_without_evaluations": curr_units_1st_without_eval,
        "Curricular_units_2nd_sem_credited": curr_units_2nd_credited,
        "Curricular_units_2nd_sem_enrolled": curr_units_2nd_enrolled,
        "Curricular_units_2nd_sem_evaluations": curr_units_2nd_evaluations,
        "Curricular_units_2nd_sem_approved": curr_units_2nd_approved,
        "Curricular_units_2nd_sem_grade": curr_units_2nd_grade,
        "Curricular_units_2nd_sem_without_evaluations": curr_units_2nd_without_eval,
        "Unemployment_rate": unemployment_rate,
        "Inflation_rate": inflation_rate,
        "GDP": gdp
    }])

    # Lakukan prediksi
    prediction = model.predict(input_data)
    result = "Dropout" if prediction[0] == 1 else "Graduated"

    st.write(f"**Hasil Prediksi:** {result}")

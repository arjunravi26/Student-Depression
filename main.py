import streamlit as st
import pickle
from PIL import Image



def main():
    st.title(":blue[STUDENT DEPRESSION PREDICTION]")
    st.sidebar.title("Checking Depression Among Student")
    st.sidebar.info("Depressed Students")
    image = Image.open('dpimg.jpg')
    st.image(image,width=500)

    Gender = st.radio("select gender",['Male','Female'],key="type_students")
    if Gender=='Female':
        gend=0
    else:
        gend=1
    Age=st.text_input("Age","Type here")
    cities=['Visakhapatnam', 'Bangalore', 'Srinagar', 'Varanasi', 'Jaipur',
            'Pune', 'Thane', 'Chennai', 'Nagpur', 'Nashik', 'Vadodara',
            'Kalyan', 'Rajkot', 'Ahmedabad', 'Kolkata', 'Mumbai', 'Lucknow',
            'Indore', 'Surat', 'Ludhiana', 'Bhopal', 'Meerut', 'Agra',
            'Ghaziabad', 'Hyderabad', 'Vasai-Virar', 'Kanpur', 'Patna',
            'Faridabad', 'Delhi', 'Saanvi', 'M.Tech', 'Bhavna', 'Less Delhi',
            'City', '3.0', 'Less than 5 Kalyan', 'Mira', 'Harsha', 'Vaanya',
            'Gaurav', 'Harsh', 'Reyansh', 'Kibara', 'Rashi', 'ME', 'M.Com',
            'Nalyan', 'Mihir', 'Nalini', 'Nandini', 'Khaziabad']
    City=st.selectbox("Enter City",cities)
    cit=cities.index(City)
    pro=['Student', 'Civil Engineer', 'Architect', 'UX/UI Designer',
         'Digital Marketer', 'Content Writer', 'Educational Consultant',
         'Teacher', 'Manager', 'Chef', 'Doctor', 'Lawyer', 'Entrepreneur','Pharmacist']
    Profession=st.selectbox("Enter Profession",pro)
    prof=pro.index(Profession)
    Academic_Pressure=st.number_input("Enter pressure range between 1 and 5")
    work=['Yes','No']
    Work_Pressure=st.selectbox("Are you suffering work pressure",work)
    if Work_Pressure == 1:
        Work_Pressure = 1
    else:
        Work_Pressure = 0
    CGPA=st.number_input("Enter your CGPA")
    Study_Satisfaction=st.number_input("Are you satisfied in your studies,if yes-rate(1-5)")
    job=['Yes','No']
    Job_Satisfaction = st.selectbox("Are you satisfied in your job",job)
    if Job_Satisfaction == 1:
        Job_Satisfaction = 1
    else:
        Job_Satisfaction = 0
    Sleep_Duration=st.number_input("Enter sleeping Hours")
    Dietary_Habits=['Healthy','Moderate','Unhealthy']
    Habits=st.selectbox("Select your dietary habits",Dietary_Habits)
    Degree=['B.Pharm', 'BSc', 'BA', 'BCA', 'M.Tech', 'PhD', 'Class 12', 'B.Ed',
            'LLB', 'BE', 'M.Ed', 'MSc', 'BHM', 'M.Pharm', 'MCA', 'MA', 'B.Com',
            'MD', 'MBA', 'MBBS', 'M.Com', 'B.Arch', 'LLM', 'B.Tech', 'BBA',
            'ME', 'MHM', 'Others']
    course=st.selectbox("Select your Degree",Degree)
    cor=Degree.index(course)
    Have_you_ever_had_suicidal_thoughts=['Yes','No']
    suicide=st.selectbox("Have you ever had suicidal thoughts",Have_you_ever_had_suicidal_thoughts)
    sui=Have_you_ever_had_suicidal_thoughts.index(suicide)
    Work_Study_Hours=st.number_input("Enter study hours")
    Financial_Stress=st.number_input("Enter Financial stress range between 1 and 5")
    Mental=['No','Yes']
    Family_History_of_Mental_Illness=st.selectbox("Is there any family history of mental illness?",Mental)

   
    pred=st.button("PREDICT")
    if pred:
        try:
            print(Gender)
            Gender = pickle.load(open('artifacts/Gender.sav','rb')).transform([Gender])[0]
            Habits = pickle.load(open('artifacts/Dietary Habits.sav','rb')).transform([Habits])[0]
            City = pickle.load(open('artifacts/City.sav','rb')).transform([City])[0]
            Profession = pickle.load(open('artifacts/Profession.sav','rb')).transform([Profession])[0]
            suicide = pickle.load(open('artifacts/Have you ever had suicidal thoughts .sav','rb')).transform([suicide])[0]
            course = pickle.load(open('artifacts/Degree.sav','rb')).transform([course])[0]
            Family_History_of_Mental_Illness = pickle.load(open('artifacts/Family History of Mental Illness.sav','rb')).transform([Family_History_of_Mental_Illness])[0]
            scaler = pickle.load(open("artifacts/scaler.sav",'rb'))
            model = pickle.load(open("artifacts/gb1model.sav",'rb'))
            features = [
                Gender,Age,City,Profession,Academic_Pressure,Work_Pressure,CGPA,Study_Satisfaction,Job_Satisfaction,
                Sleep_Duration,Habits,course,suicide,Work_Study_Hours,
                Financial_Stress,Family_History_of_Mental_Illness
                ]
            print(features)
            features_scaled = scaler.transform([features])
            print(features_scaled)
            prediction = model.predict(features_scaled)

            if prediction[0] == 1:
                st.error("Student facing depression")
            else:
                st.success("Student has no depression")
        except FileNotFoundError as e:
            st.error(f"Required file not found: {e}")
        except Exception as e:
            st.error(f"An error occurred: {e}")

main()
# Baseera: Early Autism Risk Screening and Specialist Referral System

![Flask](https://img.shields.io/badge/Flask-Backend-blue)
![Machine Learning](https://img.shields.io/badge/ML-RandomForest-green)
![Database](https://img.shields.io/badge/Supabase-Database-orange)
![Status](https://img.shields.io/badge/Status-Graduation%20Project-purple)

---

## Abstract

Baseera is a web-based intelligent screening system designed to support the early detection of Autism Spectrum Disorder (ASD) in toddlers.

The system uses a Machine Learning model (Random Forest) to analyze behavioral questionnaire responses submitted by parents and provide an initial risk assessment.

It also connects parents with specialists to ensure early follow-up and intervention.

---

## Problem Statement

Early detection of Autism Spectrum Disorder is often delayed due to the lack of accessible screening tools and limited early evaluation resources.

This project addresses this issue by providing:

- Early automated screening using AI
- Easy access for parents
- Direct referral to specialists
- Centralized case management system

---

## Project Objectives

- Support early detection of ASD indicators
- Provide AI-based risk prediction
- Assist parents with easy screening process
- Connect parents with specialists
- Manage and track screening cases
- Improve early intervention accessibility

---

## System Features

### Parent Module
- User registration and login
- Child screening questionnaire
- Instant AI prediction results
- Risk level classification
- Assign specialist doctor
- View screening history

### Specialist Module
- Secure login
- View assigned cases
- Review screening details
- Accept / manage cases
- Track patient progress

### Admin Module
- Manage users (parents & specialists)
- View all screenings
- Assign doctors to cases
- Monitor system activity

---

## Machine Learning Component

The system uses a **Random Forest Classifier** trained on autism screening data.

### Input Features
- 10 behavioral questions
- Child age
- Gender
- Jaundice history
- Family history of ASD

### Output
- Risk prediction (At Risk / Not At Risk)
- Confidence score
- Risk level (Low / Medium / High)

---

## Dataset

**Toddler Autism Dataset (July 2018)**  
Used for training and evaluating the machine learning model.

---

## Technology Stack

### Backend
- Python
- Flask
- Supabase

### Frontend
- HTML5
- CSS3
- JavaScript

### Machine Learning
- Scikit-learn
- Random Forest
- NumPy
- Pandas

### Deployment
- Vercel

---

## System Architecture

1. Parent submits screening form
2. Flask API processes input
3. ML model generates prediction
4. Results stored in Supabase database
5. Parent assigns specialist
6. Specialist receives and reviews case
7. Admin monitors system

---

## Project Structure

Baseera/
│
├── app.py
├── requirements.txt
├── runtime.txt
├── vercel.json
│
├── templates/
│   ├── Index.html
│   ├── Parent.html
│   ├── Specialist.html
│   └── Admin.html
│
├── static/
│   └── logo.png
│
├── rf_model.pkl
├── model_meta.json
│
├── Train_model.py
├── verify_model.py
│
└── Toddler Autism Dataset July 2018 (1).csv── Toddler Autism Dataset July 2018 (1).csv

---

## API Endpoints

### Prediction
- `POST /api/predict`  
  Generate autism risk prediction

### History
- `GET /api/history/<email>`  
  Get parent screening history

### Specialist
- `GET /api/specialist/<doctor_id>/children`  
  Get assigned cases

### Assign Doctor
- `POST /api/assign-doctor`  
  Assign specialist to case

### Accept Case
- `POST /api/accept-case`  
  Update case status

---

## Ethical Considerations

This system is a **screening support tool only** and does not provide medical diagnosis.

All results should be reviewed by qualified healthcare professionals.

---

## Limitations

- Not a medical diagnostic system
- Depends on dataset quality
- Requires internet connection
- Works only with structured questionnaire input

---

## Future Enhancements

- Deep learning integration
- Mobile application version
- Real-time notifications
- Appointment booking system
- Advanced analytics dashboard
- Multilingual support

---

## Authors

- Fajer  
- Nora  
- Retaj  
- Latifah  
- Khulood  
- Raneem  
- Bayan  

---

## Academic Information

College of Computer Science and Engineering  
Data Science Program  
Bachelor’s Graduation Project (2026)

### Supervisor
Dr. Malak Al-Houti

-- Updated schema for Supabase SQL Editor

-- Doctors table for storing medical professionals
CREATE TABLE IF NOT EXISTS doctors (
    id SERIAL PRIMARY KEY,
    email TEXT UNIQUE NOT NULL,
    password TEXT NOT NULL,  -- Changed from BYTEA to TEXT for base64 encoded passwords
    name TEXT NOT NULL,
    specialty TEXT,
    phone TEXT,
    profile_image TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Patients table for storing patient information
CREATE TABLE IF NOT EXISTS patients (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    age INTEGER NOT NULL,
    gender TEXT NOT NULL,
    medical_history TEXT,
    additional_notes TEXT,
    last_visit TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    doctor_id INTEGER REFERENCES doctors(id) NOT NULL,
    status TEXT DEFAULT 'Pending'
);

-- Tests table for storing medical tests and results
CREATE TABLE IF NOT EXISTS tests (
    id SERIAL PRIMARY KEY,
    test_type TEXT NOT NULL,
    image_path TEXT,
    result TEXT,
    result_image_path TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    patient_id INTEGER REFERENCES patients(id) NOT NULL,
    doctor_notes TEXT,
    report_path TEXT
);

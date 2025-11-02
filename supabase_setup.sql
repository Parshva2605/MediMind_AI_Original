-- Drop existing tables if they exist
DROP TABLE IF EXISTS tests;
DROP TABLE IF EXISTS patients;
DROP TABLE IF EXISTS doctors;

-- Doctors table for storing medical professionals
CREATE TABLE doctors (
    id SERIAL PRIMARY KEY,
    email TEXT UNIQUE NOT NULL,
    password TEXT NOT NULL,  -- TEXT for base64 encoded passwords
    name TEXT NOT NULL,
    specialty TEXT,
    phone TEXT,
    profile_image TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Patients table for storing patient information
CREATE TABLE patients (
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
CREATE TABLE tests (
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

-- Disable Row Level Security (for development)
ALTER TABLE doctors DISABLE ROW LEVEL SECURITY;
ALTER TABLE patients DISABLE ROW LEVEL SECURITY;
ALTER TABLE tests DISABLE ROW LEVEL SECURITY;

-- Grant access to tables
GRANT ALL PRIVILEGES ON doctors TO anon, authenticated, service_role;
GRANT ALL PRIVILEGES ON patients TO anon, authenticated, service_role;
GRANT ALL PRIVILEGES ON tests TO anon, authenticated, service_role;
GRANT USAGE, SELECT ON SEQUENCE doctors_id_seq TO anon, authenticated, service_role;
GRANT USAGE, SELECT ON SEQUENCE patients_id_seq TO anon, authenticated, service_role;
GRANT USAGE, SELECT ON SEQUENCE tests_id_seq TO anon, authenticated, service_role;

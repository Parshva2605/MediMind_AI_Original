"""
Supabase Setup Script - MediMind AI

This script helps set up the necessary tables in Supabase for the MediMind AI application.
Before running this script, make sure to update the .env file with your Supabase URL and API key.
"""

import os
from dotenv import load_dotenv
from supabase import create_client

# Load environment variables
load_dotenv()

# Initialize Supabase client
supabase_url = os.getenv('SUPABASE_URL')
supabase_key = os.getenv('SUPABASE_KEY')

if not supabase_url or not supabase_key:
    print("Error: Supabase credentials not found. Please update your .env file.")
    exit(1)

supabase = create_client(supabase_url, supabase_key)

# SQL commands to create tables
create_doctors_table = """
CREATE TABLE IF NOT EXISTS doctors (
    id SERIAL PRIMARY KEY,
    email TEXT UNIQUE NOT NULL,
    password BYTEA NOT NULL,
    name TEXT NOT NULL,
    specialty TEXT,
    phone TEXT,
    profile_image TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);
"""

create_patients_table = """
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
"""

create_tests_table = """
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
"""

# Execute SQL commands
try:
    print("Creating doctors table...")
    supabase.table("doctors").query("count").execute()
    print("Doctors table already exists")
except Exception:
    print("Creating doctors table...")
    response = supabase.rpc('exec_sql', {'query': create_doctors_table}).execute()
    print("Doctors table created")

try:
    print("Creating patients table...")
    supabase.table("patients").query("count").execute()
    print("Patients table already exists")
except Exception:
    print("Creating patients table...")
    response = supabase.rpc('exec_sql', {'query': create_patients_table}).execute()
    print("Patients table created")

try:
    print("Creating tests table...")
    supabase.table("tests").query("count").execute()
    print("Tests table already exists")
except Exception:
    print("Creating tests table...")
    response = supabase.rpc('exec_sql', {'query': create_tests_table}).execute()
    print("Tests table created")

print("Database setup complete!")

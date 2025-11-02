#!/bin/bash
echo "======================================"
echo "MediMind AI - Setup and Run Script"
echo "======================================"
echo

echo "Installing core dependencies..."
pip install Flask Flask-SQLAlchemy Werkzeug

echo
echo "Core installation complete!"
echo

read -p "Would you like to install AI and visualization dependencies? (y/n) " INSTALL_AI

if [[ $INSTALL_AI == "y" || $INSTALL_AI == "Y" ]]; then
    echo
    echo "Installing AI and visualization dependencies..."
    pip install tensorflow numpy Pillow opencv-python matplotlib fpdf
    echo "AI dependencies installed!"
else
    echo
    echo "Skipping AI dependencies."
    echo "Note: AI features will be disabled."
fi

echo
echo "======================================"
echo "Starting MediMind AI Application"
echo "======================================"
echo
echo "Access the application at: http://localhost:5000"
echo "Press Ctrl+C to stop the server"
echo

python app.py

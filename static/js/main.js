// Main JavaScript file for MediMind AI

// Handle sidebar toggle for smaller screens
document.addEventListener('DOMContentLoaded', function() {
    // Set up theme
    const savedTheme = localStorage.getItem('theme') || 'light';
    setTheme(savedTheme);
    
    // Update theme selector if on settings page
    const themeSelector = document.getElementById('themeSelector');
    if (themeSelector) {
        themeSelector.value = savedTheme;
    }
});

// Theme switcher function
function setTheme(themeName) {
    const body = document.body;
    
    // Remove any existing theme classes
    body.classList.remove('theme-light', 'theme-dark');
    
    // Add the selected theme
    body.classList.add(`theme-${themeName}`);
    
    // Save to local storage
    localStorage.setItem('theme', themeName);
}

// Handle image uploads
function handleImageUpload(inputElement, previewElement) {
    const file = inputElement.files[0];
    if (!file) return;
    
    // Validate file is an image
    if (!file.type.match('image.*')) {
        alert('Please select an image file');
        return;
    }
    
    const reader = new FileReader();
    reader.onload = function(e) {
        previewElement.src = e.target.result;
        previewElement.style.display = 'block';
    };
    reader.readAsDataURL(file);
}

// Search functionality
function filterPatients(searchTerm) {
    const patientCards = document.querySelectorAll('.patient-card');
    
    patientCards.forEach(card => {
        const patientName = card.querySelector('.card-title').textContent.toLowerCase();
        const patientInfo = card.querySelector('.card-body').textContent.toLowerCase();
        
        if (patientName.includes(searchTerm.toLowerCase()) || patientInfo.includes(searchTerm.toLowerCase())) {
            card.style.display = '';
        } else {
            card.style.display = 'none';
        }
    });
}

// Initialize search functionality if search input exists
const searchInput = document.getElementById('searchPatient');
if (searchInput) {
    searchInput.addEventListener('input', function() {
        filterPatients(this.value);
    });
}

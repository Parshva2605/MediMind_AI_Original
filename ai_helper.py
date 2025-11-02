import json
import requests
import time

def generate_ai_summary(patient_data, test_data, result_data):
    """
    Generate an AI summary using the local Ollama LLM (deepseek-r1:7b).
    
    Args:
        patient_data: Dictionary containing patient information
        test_data: Dictionary containing test information
        result_data: Dictionary containing test results
    
    Returns:
        A string containing the AI-generated summary or None if generation failed
    """
    try:
        # Ollama API URL
        api_url = "http://localhost:11434/api/generate"
        
        # Extract relevant patient information
        patient_name = patient_data.get('name', 'Unknown')
        patient_age = patient_data.get('age', 'Unknown')
        patient_gender = patient_data.get('gender', 'Unknown')
        medical_history = patient_data.get('medical_history', 'None')
        
        # Extract test information
        test_type = test_data.get('test_type', 'Unknown').replace('_', ' ').title()
        
        # Format the prompt for the LLM
        prompt = f"""You are a medical AI assistant helping to generate a summary for a medical report.

Patient Information:
- Name: {patient_name}
- Age: {patient_age}
- Gender: {patient_gender}
- Medical History: {medical_history}

Test Information:
- Test Type: {test_type}
- Results: {json.dumps(result_data, indent=2)}

Please provide a concise medical summary (around 300 words) that includes:
1. An interpretation of the test results
2. Potential concerns or observations
3. Recommended next steps or follow-up tests
4. Lifestyle or treatment recommendations based on these findings

Your summary should be professional, medically accurate, and helpful for both the doctor and patient to understand the implications of the test results.
"""
        
        # Prepare request to Ollama API
        payload = {
            "model": "deepseek-r1:7b",
            "prompt": prompt,
            "stream": False,
            "temperature": 0.7,
            "max_tokens": 1000
        }
        
        # Try to connect to Ollama (with timeout)
        try:
            response = requests.post(api_url, json=payload, timeout=30)
            response.raise_for_status()  # Raise exception for HTTP errors
            
            result = response.json()
            summary = result.get('response', '')
            
            # Clean up the summary text
            summary = summary.strip()
            
            return summary
        
        except requests.exceptions.RequestException as e:
            print(f"Error connecting to Ollama API: {str(e)}")
            return generate_fallback_summary(patient_data, test_data, result_data)
    
    except Exception as e:
        print(f"Error generating AI summary: {str(e)}")
        return generate_fallback_summary(patient_data, test_data, result_data)


def generate_fallback_summary(patient_data, test_data, result_data):
    """Generate a fallback summary when Ollama is not available"""
    
    test_type = test_data.get('test_type', '').replace('_', ' ').title()
    
    # Check if this is a chest X-ray test
    if 'chest_xray' in test_data.get('test_type', '').lower():
        # Try to extract prediction information
        detected_count = 0
        detected_conditions = []
        threshold_used = 0.35  # Default threshold for best_chest_model.h5
        
        try:
            if isinstance(result_data, str):
                result_data = json.loads(result_data)
            
            # Get threshold if available
            if 'threshold_used' in result_data:
                threshold_used = result_data['threshold_used']
            
            # Get detected conditions
            if 'above_threshold' in result_data:
                detected_conditions = list(result_data['above_threshold'].items())
                detected_count = len(detected_conditions)
            elif 'total_detected' in result_data:
                detected_count = result_data['total_detected']
            
            # Build summary based on results
            summary = f"Medical Summary for {patient_data.get('name', 'Patient')}\n\n"
            summary += f"Test Type: Chest X-Ray Analysis (14-Disease Detection)\n"
            summary += f"Model: Best Chest Model (94.90% Accuracy)\n"
            summary += f"Detection Threshold: {threshold_used * 100:.0f}%\n\n"
            
            if detected_count == 0:
                summary += "FINDINGS:\n"
                summary += "The chest X-ray analysis shows NO significant abnormalities detected above the threshold. "
                summary += "All 14 tested conditions (Pneumonia, Atelectasis, Cardiomegaly, Effusion, Mass, Nodule, "
                summary += "Pneumothorax, Consolidation, Edema, Emphysema, Fibrosis, Pleural Thickening, Infiltration, and Hernia) "
                summary += "are within normal ranges.\n\n"
                summary += "INTERPRETATION:\n"
                summary += "The X-ray appears normal with no immediate concerns identified by the AI analysis.\n\n"
                summary += "RECOMMENDATIONS:\n"
                summary += "1. Continue routine health monitoring\n"
                summary += "2. Maintain healthy lifestyle habits\n"
                summary += "3. Schedule regular check-ups as recommended by your physician\n"
                summary += "4. If symptoms develop, consult your doctor immediately\n"
            else:
                summary += f"FINDINGS:\n"
                summary += f"The AI analysis detected {detected_count} condition(s) above the {threshold_used * 100:.0f}% threshold:\n\n"
                
                for condition, probability in sorted(detected_conditions, key=lambda x: x[1], reverse=True):
                    summary += f"• {condition}: {probability * 100:.1f}% probability\n"
                
                summary += "\nINTERPRETATION:\n"
                summary += "These findings indicate potential abnormalities that require medical attention. "
                summary += "The AI model has identified patterns consistent with the above conditions. "
                summary += "Please note that AI analysis should be confirmed by a qualified radiologist.\n\n"
                
                summary += "RECOMMENDATIONS:\n"
                summary += "1. IMMEDIATE: Consult with a radiologist for confirmation\n"
                summary += "2. Further diagnostic tests may be required based on clinical correlation\n"
                summary += "3. Follow-up imaging may be necessary\n"
                summary += "4. Treatment plan should be developed based on confirmed diagnosis\n"
                summary += "5. Monitor symptoms and report any changes to your healthcare provider\n"
            
            summary += "\nIMPORTANT NOTE:\n"
            summary += "This AI-assisted analysis is a screening tool and should NOT replace professional medical judgment. "
            summary += "All findings must be reviewed and confirmed by a qualified healthcare professional.\n"
            
            return summary
            
        except Exception as e:
            print(f"Error creating detailed fallback summary: {e}")
            # Return basic summary if detailed processing fails
            return f"Chest X-Ray analysis completed for {patient_data.get('name', 'Patient')}. Please review the detailed results above. This is an AI-assisted screening tool and all findings should be confirmed by a qualified radiologist."
    
    # Check if this is a lung cancer test
    if 'lung_cancer' in test_data.get('test_type', '').lower():
        try:
            if isinstance(result_data, str):
                result_data = json.loads(result_data)
            
            prediction = result_data.get('prediction', 'Unknown')
            confidence = result_data.get('confidence', 0) * 100
            probabilities = result_data.get('probabilities', {})
            
            summary = f"Medical Summary for {patient_data.get('name', 'Patient')}\n\n"
            summary += f"Test Type: Lung Cancer CT Scan Analysis\n"
            summary += f"Model: stage2_best.h5 (96.8% Accuracy)\n"
            summary += f"Classification: Binary (Malignant vs Non-malignant)\n\n"
            
            summary += f"FINDINGS:\n"
            summary += f"The AI analysis of the CT scan has classified the lung tissue as: **{prediction}**\n"
            summary += f"Confidence Level: {confidence:.1f}%\n\n"
            
            if probabilities:
                summary += "Detailed Probability Breakdown:\n"
                for class_name, prob in probabilities.items():
                    summary += f"• {class_name}: {prob * 100:.1f}%\n"
                summary += "\n"
            
            if prediction == "Malignant":
                summary += "INTERPRETATION:\n"
                summary += "The AI model has detected patterns consistent with malignant lung tissue. "
                summary += "This indicates a high likelihood of cancerous cells in the analyzed CT scan. "
                summary += "Immediate medical attention is strongly recommended.\n\n"
                
                summary += "RECOMMENDATIONS:\n"
                summary += "1. **URGENT**: Consult with an oncologist immediately\n"
                summary += "2. **Biopsy**: Tissue biopsy recommended for definitive diagnosis\n"
                summary += "3. **Staging**: Complete cancer staging workup (PET scan, brain MRI)\n"
                summary += "4. **Treatment Planning**: Discuss surgery, chemotherapy, radiation options\n"
                summary += "5. **Second Opinion**: Consider consultation at a cancer center\n"
                summary += "6. **Family Support**: Engage family support and counseling services\n"
                summary += "7. **Clinical Trials**: Explore available clinical trial options\n"
            else:
                summary += "INTERPRETATION:\n"
                summary += "The AI model has classified the lung tissue as non-malignant. "
                summary += "This suggests the analyzed tissue is either benign or normal. "
                summary += "However, clinical correlation and expert review are essential.\n\n"
                
                summary += "RECOMMENDATIONS:\n"
                summary += "1. **Radiologist Review**: Have results reviewed by a qualified radiologist\n"
                summary += "2. **Clinical Correlation**: Correlate with symptoms and clinical presentation\n"
                summary += "3. **Follow-up**: Regular monitoring if benign findings present\n"
                summary += "4. **Additional Tests**: Consider bronchoscopy if symptoms persist\n"
                summary += "5. **Lifestyle**: Smoking cessation if applicable\n"
                summary += "6. **Monitoring**: Schedule follow-up CT scans as recommended\n"
            
            summary += "\nIMPORTANT DISCLAIMER:\n"
            summary += "This AI analysis is a diagnostic aid and NOT a definitive diagnosis. "
            summary += "All findings must be confirmed by a board-certified radiologist and oncologist. "
            summary += "Treatment decisions should be made only after comprehensive evaluation by qualified healthcare professionals.\n"
            
            return summary
            
        except Exception as e:
            print(f"Error creating lung cancer summary: {e}")
            return f"Lung Cancer CT scan analysis completed for {patient_data.get('name', 'Patient')}. Classification: {result_data.get('prediction', 'Unknown')}. Please consult with an oncologist for detailed interpretation."
    
    # Generic fallback for other test types  
    return f"""Medical Summary

The {test_type} test has been completed successfully. Based on the results, a thorough evaluation by a specialist is recommended to determine the appropriate next steps.

General Recommendations:
1. Schedule a follow-up appointment to discuss these results in detail
2. Continue monitoring any symptoms related to the condition
3. Maintain a healthy lifestyle with proper diet and exercise
4. Follow any medication regimens previously prescribed

This is a computer-assisted analysis and should be confirmed by a qualified healthcare professional. Regular follow-ups are essential for proper management of the patient's condition.
"""

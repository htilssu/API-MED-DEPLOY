# User Disease Information System Documentation

## Overview
This system provides comprehensive management of user disease information, extending the existing diagnosis system with detailed tracking capabilities.

## Models

### Basic Diagnosis System (Existing)
- **DiagnoseModel**: Basic diagnosis information with disease results
- **CreateDiagnoseModel**: For creating new diagnosis records

### Enhanced User Disease System (New)
- **UserDiseaseModel**: Comprehensive disease information with detailed tracking
- **CreateUserDiseaseModel**: For creating new detailed disease records
- **UpdateUserDiseaseModel**: For updating existing disease information
- **UserDiseaseHistoryModel**: Overall disease history for a user

## Key Features

### Disease Status Tracking
- `ACTIVE`: Currently suffering from the disease
- `RECOVERED`: Recovered from the disease
- `CHRONIC`: Chronic condition
- `UNDER_TREATMENT`: Currently under treatment

### Severity Levels
- `MILD`: Minor symptoms
- `MODERATE`: Moderate symptoms
- `SEVERE`: Serious symptoms
- `CRITICAL`: Life-threatening condition

### Comprehensive Information
- Disease name and ICD-10 code
- Symptoms and detailed description
- Treatment plans and medications
- Doctor and hospital information
- AI diagnosis integration
- Follow-up scheduling
- Personal notes

## API Endpoints

### Basic Diagnosis (Existing)
- `POST /api/create-diagnosis` - Create basic diagnosis
- `GET /api/get-diagnosis/{diagnosis_id}` - Get diagnosis by ID
- `GET /api/user-diagnoses/{user_id}` - Get user's diagnoses
- `DELETE /api/delete-diagnosis/{diagnosis_id}` - Delete diagnosis

### Enhanced User Disease Management (New)
- `POST /api/user-diseases` - Create detailed disease information
- `GET /api/user-diseases/{disease_id}` - Get disease information by ID
- `GET /api/users/{user_id}/diseases` - Get all user diseases (with status filtering)
- `PUT /api/user-diseases/{disease_id}` - Update disease information
- `DELETE /api/user-diseases/{disease_id}` - Delete disease information
- `GET /api/users/{user_id}/disease-history` - Get complete disease history
- `GET /api/users/{user_id}/diseases/search` - Search user diseases
- `POST /api/users/{user_id}/diseases/from-ai-diagnosis` - Create from AI diagnosis
- `PATCH /api/user-diseases/{disease_id}/status` - Update disease status
- `GET /api/users/{user_id}/disease-stats` - Get disease statistics

## Usage Examples

### Creating a Disease Record
```json
POST /api/user-diseases
{
    "userId": "user123",
    "diseaseName": "Viêm da cơ địa",
    "diseaseCode": "L20.9",
    "status": "active",
    "severity": "moderate",
    "symptoms": ["Ngứa", "Đỏ da", "Khô da"],
    "treatment": "Dùng kem dưỡng ẩm",
    "medications": ["Cetirizine 10mg"],
    "doctorName": "BS. Nguyễn Văn A",
    "hospitalName": "Bệnh viện Da liễu TP.HCM"
}
```

### Filtering by Status
```
GET /api/users/user123/diseases?status=active
```

### Searching Diseases
```
GET /api/users/user123/diseases/search?q=viêm da
```

### Creating from AI Diagnosis
```
POST /api/users/user123/diseases/from-ai-diagnosis?diagnosis_key=redis_key&disease_names=Disease1,Disease2&confidence=0.85
```

## Database Collections
- `diagnoses` - Basic diagnosis records (existing)
- `user_diseases` - Detailed user disease information (new)

## Integration with AI System
The system integrates with the existing AI diagnosis pipeline:
- AI diagnosis results can be automatically converted to user disease records
- Confidence scores and diagnosis keys are preserved
- Supports batch creation from multiple disease predictions

## Benefits
1. **Comprehensive Tracking**: Detailed disease information beyond basic diagnosis
2. **Status Management**: Track disease progression and treatment outcomes
3. **Medical History**: Complete medical history for each user
4. **AI Integration**: Seamless integration with AI diagnosis results
5. **Search & Filter**: Easy retrieval of specific disease information
6. **Statistics**: Disease trends and statistics for users
7. **Professional Integration**: Store doctor and hospital information
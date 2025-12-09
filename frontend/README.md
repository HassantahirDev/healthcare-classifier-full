# Frontend - Healthcare Disease Classifier

This is my React-based web application for the disease prediction interface.

## Quick Start

### Installation

```bash
cd frontend
npm install
```

### Configuration

Create a `.env` file:

```
REACT_APP_API_URL=http://localhost:5000
```

### Running the App

```bash
npm start
```

The app will open at: `http://localhost:3000`

## Features

### 1. Symptom Input
- Text area for entering symptoms
- Real-time validation
- Quick example buttons
- Clear and reset functionality

### 2. Prediction Results
- Results from all 4 models displayed simultaneously
- Confidence scores with visual indicators
- Color-coded confidence levels:
  - Green (≥80%): High confidence
  - Orange (60-79%): Medium confidence
  - Red (<60%): Low confidence

### 3. Model Information
- Display of all available models
- Performance metrics (accuracy)
- Real-time API status checking

### 4. User Experience
- Modern, gradient-based UI
- Responsive design (mobile-friendly)
- Loading states and animations
- Error handling and messaging
- Keyboard shortcuts (Enter to predict)

## Components

### App.js
Main application component that:
- Manages application state
- Handles API communication
- Controls user interactions
- Displays health status

### PredictionResults.js
Displays prediction results:
- Disease predictions from all models
- Confidence scores with progress bars
- Color-coded confidence indicators
- Analyzed symptoms display

### ModelInfo.js
Shows model information:
- Available models list
- Performance metrics
- Model accuracy comparison

## Project Structure

```
frontend/
├── public/
│   └── index.html
├── src/
│   ├── components/
│   │   ├── PredictionResults.js
│   │   └── ModelInfo.js
│   ├── services/
│   │   └── api.js
│   ├── App.js
│   ├── App.css
│   └── index.js
├── package.json
└── README.md
```

## API Integration

### Service Layer (api.js)

I created a service layer to handle all API calls:

```javascript
import { predictAll, predictSingle, getModels, checkHealth } from './services/api';

// Predict with all models
const results = await predictAll(symptoms);

// Predict with specific model
const result = await predictSingle(symptoms, 'lstm');

// Get available models
const models = await getModels();

// Check API health
const status = await checkHealth();
```

### API Functions

1. **predictAll(symptoms)**: Get predictions from all models
2. **predictSingle(symptoms, modelName)**: Get prediction from one model
3. **getModels()**: Fetch available models and performance
4. **getClasses()**: Get list of disease classes
5. **checkHealth()**: Check API availability

## User Flow

1. **Landing**: User sees the main interface with symptom input
2. **Input**: User enters symptoms or selects an example
3. **Predict**: User clicks "Predict Disease" button
4. **Loading**: Loading animation displays while processing
5. **Results**: Predictions from all 4 models are shown
6. **Compare**: User can compare different model predictions
7. **Reset**: User can clear and start over

## Styling

### Design System

**Colors:**
- Primary: `#667eea` (Purple)
- Secondary: `#764ba2` (Deep Purple)
- Success: `#4caf50` (Green)
- Warning: `#ff9800` (Orange)
- Error: `#f44336` (Red)

**Typography:**
- Font: System fonts (-apple-system, BlinkMacSystemFont, Segoe UI)
- Headings: Bold, various sizes
- Body: Regular weight

**Layout:**
- Max width: 1200px
- Responsive grid
- Card-based design
- Gradient backgrounds

### Responsive Design

- **Desktop**: Full grid layout, multi-column
- **Tablet**: Adjusted grid, 2 columns
- **Mobile**: Single column, stacked layout

## Testing

### Manual Testing

1. Start the backend server
2. Start the frontend app
3. Test features:
   - Enter symptoms manually
   - Click example symptoms
   - Make predictions
   - Clear and reset
   - Check error handling

### Example Symptoms

```
- "fever, cough, and difficulty breathing"
- "severe headache, nausea, and sensitivity to light"
- "chest pain, shortness of breath"
- "joint pain, fatigue, and muscle weakness"
```

## Build & Deploy

### Production Build

```bash
npm run build
```

Creates optimized production build in `build/` directory.

### Deployment Options

#### Netlify
```bash
npm run build
# Deploy 'build' directory to Netlify
```

#### Vercel
```bash
npm run build
# Deploy 'build' directory to Vercel
```

#### GitHub Pages
```bash
npm run build
# Deploy to gh-pages branch
```

### Environment Variables

For production, update `.env`:

```
REACT_APP_API_URL=https://your-api-domain.com
```

## Dependencies

### Core
- **react**: ^18.2.0 - UI library
- **react-dom**: ^18.2.0 - DOM rendering
- **react-scripts**: 5.0.1 - Build scripts

### HTTP Client
- **axios**: ^1.4.0 - API requests

### Utilities
- **web-vitals**: ^2.1.4 - Performance metrics

## Configuration

### Package.json Scripts

```json
{
  "start": "react-scripts start",
  "build": "react-scripts build",
  "test": "react-scripts test",
  "eject": "react-scripts eject"
}
```

### Browser Support

- Chrome (latest)
- Firefox (latest)
- Safari (latest)
- Edge (latest)
- Mobile browsers

## Troubleshooting

### API Connection Issues

**Problem**: Cannot connect to backend
**Solution**:
- Ensure backend is running on port 5000
- Check `.env` file has correct API URL
- Verify CORS is enabled on backend
- Check browser console for errors

### Build Errors

**Problem**: npm install fails
**Solution**:
- Delete `node_modules/` and `package-lock.json`
- Run `npm install` again
- Check Node.js version (16+)

### Blank Page

**Problem**: App shows blank page
**Solution**:
- Check browser console for errors
- Verify all components are imported correctly
- Ensure `public/index.html` has `<div id="root">`

## Customization

### Changing Colors

Edit `App.css`:

```css
/* Primary gradient */
background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);

/* Success color */
color: #4caf50;
```

### Adding New Features

1. Create new component in `src/components/`
2. Import in `App.js`
3. Add to render method
4. Update styling in `App.css`

## Notes

This frontend requires the backend API to be running for full functionality. I designed it to work seamlessly with the Flask backend I created.

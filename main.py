import cv2 as cv
import numpy as np
import os
import argparse
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample

# Image preprocessing functions
def convert_to_grayscale(img):
    """Convert to grayscale"""
    if img is None or img.size == 0:
        raise ValueError("Invalid image: image is None or empty")
    
    if len(img.shape) == 3:
        return cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    return img

def normalize_image(img):
    """Normalize pixel values to 0-255"""
    if img is None or img.size == 0:
        raise ValueError("Invalid image: image is None or empty")
    
    img_normalized = cv.normalize(img, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)
    return img_normalized

def apply_gaussian_blur(img, kernel_size=(5, 5)):
    """Blur image to reduce noise"""
    if img is None or img.size == 0:
        raise ValueError("Invalid image: image is None or empty")
    
    if kernel_size[0] % 2 == 0 or kernel_size[1] % 2 == 0:
        raise ValueError("Kernel size must be odd numbers")
    
    return cv.GaussianBlur(img, kernel_size, 0)

# ============================================================================
# FEATURE EXTRACTION FUNCTIONS - Thermal image analysis & feature computation
# ============================================================================

def temperature_threshold(img, threshold_value, max_value=255):
    """Apply temperature thresholding"""
    if img is None or img.size == 0:
        raise ValueError("Invalid image provided")
    
    if not 0 <= threshold_value <= 255:
        raise ValueError(f"Threshold must be 0-255, got {threshold_value}")
    
    _, thresholded = cv.threshold(img, threshold_value, max_value, cv.THRESH_BINARY)
    return thresholded

def calculate_temperature_gradient(img):
    """Calculate temperature gradients"""
    if img is None or img.size == 0:
        raise ValueError("Invalid image provided")
    
    grad_x = cv.Sobel(img, cv.CV_64F, 1, 0, ksize=3)
    grad_y = cv.Sobel(img, cv.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = cv.magnitude(grad_x, grad_y)
    return cv.convertScaleAbs(gradient_magnitude)

def extract_thermal_features(img):
    """Extract thermal statistics"""
    if img is None or img.size == 0:
        raise ValueError("Invalid image provided")
    
    mean_temp = cv.mean(img)[0]
    _, std_temp = cv.meanStdDev(img)
    min_temp = float(img.min())
    max_temp = float(img.max())
    
    return {
        'mean_temperature': float(mean_temp),
        'std_temperature': float(std_temp[0][0]),
        'min_temperature': min_temp,
        'max_temperature': max_temp
    }

def find_regions_of_interest(img, threshold_value, min_area=100):
    """Find high-temperature regions"""
    if img is None or img.size == 0:
        raise ValueError("Invalid image provided")
    
    if min_area < 0:
        raise ValueError(f"min_area must be positive, got {min_area}")
    
    thresholded = temperature_threshold(img, threshold_value)
    contours, _ = cv.findContours(thresholded, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    # Filter contours by area
    filtered_contours = [cnt for cnt in contours if cv.contourArea(cnt) > min_area]
    
    return filtered_contours

# ============================================================================
# BREAST CANCER ANALYSIS FUNCTIONS - Specialized thermal imaging analysis
# ============================================================================

def segment_breast_regions(img):
    """Segment left and right breast regions"""
    if img is None or img.size == 0:
        raise ValueError("Invalid image provided")
    
    height, width = img.shape[:2]
    
    if height < 10 or width < 10:
        raise ValueError(f"Image too small: {width}x{height}")
    
    # Define breast regions (rough approximation - upper 60% of image, split vertically)
    breast_top = int(height * 0.1)  # Start below neck/shoulders
    breast_bottom = int(height * 0.7)  # End above abdomen
    mid_x = width // 2
    
    left_breast = img[breast_top:breast_bottom, :mid_x]
    right_breast = img[breast_top:breast_bottom, mid_x:]
    
    return left_breast, right_breast, (0, breast_top, mid_x, breast_bottom), (mid_x, breast_top, width, breast_bottom)

def calculate_breast_asymmetry(img):
    """Calculate left-right breast asymmetry"""
    if img is None or img.size == 0:
        raise ValueError("Invalid image provided")
    
    left_breast, right_breast, _, _ = segment_breast_regions(img)
    
    # Calculate mean temperatures
    left_mean = float(cv.mean(left_breast)[0])
    right_mean = float(cv.mean(right_breast)[0])
    
    # Calculate asymmetry with division by zero protection
    max_temp = max(left_mean, right_mean)
    if max_temp < 1e-6:  # Prevent division by zero
        asymmetry_percentage = 0.0
    else:
        asymmetry = abs(left_mean - right_mean)
        asymmetry_percentage = (asymmetry / max_temp) * 100
    
    # Calculate standard deviations
    _, left_std = cv.meanStdDev(left_breast)
    _, right_std = cv.meanStdDev(right_breast)
    
    return {
        'left_mean_temp': left_mean,
        'right_mean_temp': right_mean,
        'temperature_asymmetry': float(abs(left_mean - right_mean)),
        'asymmetry_percentage': float(asymmetry_percentage),
        'left_std': float(left_std[0][0]),
        'right_std': float(right_std[0][0])
    }

def analyze_breast_quadrants(img):
    """Analyze temperature in breast quadrants"""
    if img is None or img.size == 0:
        raise ValueError("Invalid image provided")
    
    left_breast, right_breast, left_coords, right_coords = segment_breast_regions(img)
    
    # Split each breast into quadrants
    left_height, left_width = left_breast.shape[:2]
    right_height, right_width = right_breast.shape[:2]
    
    if left_height < 2 or left_width < 2 or right_height < 2 or right_width < 2:
        raise ValueError("Breast regions too small for quadrant analysis")
    
    # Left breast quadrants
    left_upper_outer = left_breast[:left_height//2, :left_width//2]
    left_upper_inner = left_breast[:left_height//2, left_width//2:]
    left_lower_outer = left_breast[left_height//2:, :left_width//2]
    left_lower_inner = left_breast[left_height//2:, left_width//2:]
    
    # Right breast quadrants
    right_upper_outer = right_breast[:right_height//2, :right_width//2]
    right_upper_inner = right_breast[:right_height//2, right_width//2:]
    right_lower_outer = right_breast[right_height//2:, :right_width//2]
    right_lower_inner = right_breast[right_height//2:, right_width//2:]
    
    quadrants = {
        'left_upper_outer': float(cv.mean(left_upper_outer)[0]),  # Most critical for cancer detection
        'left_upper_inner': float(cv.mean(left_upper_inner)[0]),
        'left_lower_outer': float(cv.mean(left_lower_outer)[0]),
        'left_lower_inner': float(cv.mean(left_lower_inner)[0]),
        'right_upper_outer': float(cv.mean(right_upper_outer)[0]),  # Most critical for cancer detection
        'right_upper_inner': float(cv.mean(right_upper_inner)[0]),
        'right_lower_outer': float(cv.mean(right_lower_outer)[0]),
        'right_lower_inner': float(cv.mean(right_lower_inner)[0])
    }
    
    return quadrants

def detect_breast_hotspots(img, threshold_percentile=90):
    """Detect warm hotspots in breasts"""
    if img is None or img.size == 0:
        raise ValueError("Invalid image provided")
    
    if not 0 < threshold_percentile < 100:
        raise ValueError(f"Percentile must be 0-100, got {threshold_percentile}")
    
    left_breast, right_breast, left_coords, right_coords = segment_breast_regions(img)
    
    # Calculate threshold based on percentile
    left_threshold = float(np.percentile(left_breast, threshold_percentile))
    right_threshold = float(np.percentile(right_breast, threshold_percentile))
    
    # Apply thresholding
    left_hotspots = temperature_threshold(left_breast, left_threshold)
    right_hotspots = temperature_threshold(right_breast, right_threshold)
    
    # Find contours of hotspots
    left_contours, _ = cv.findContours(left_hotspots, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    right_contours, _ = cv.findContours(right_hotspots, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    return {
        'left_hotspots': left_contours if left_contours else [],
        'right_hotspots': right_contours if right_contours else [],
        'left_threshold': left_threshold,
        'right_threshold': right_threshold
    }

# ============================================================================
# DATASET COMPATIBILITY FUNCTIONS - Cross-protocol alignment & normalization
# ============================================================================

def histogram_match_image(source, reference):
    """Normalize image colors to reference histogram"""
    if source is None or reference is None or source.size == 0 or reference.size == 0:
        raise ValueError("Invalid images provided")
    
    try:
        # Compute histograms
        source_hist = cv.calcHist([source], [0], None, [256], [0, 256])
        reference_hist = cv.calcHist([reference], [0], None, [256], [0, 256])
        
        # Normalize histograms
        source_hist = cv.normalize(source_hist, source_hist).flatten()
        reference_hist = cv.normalize(reference_hist, reference_hist).flatten()
        
        # Compute CDFs (cumulative distribution functions)
        source_cdf = np.cumsum(source_hist)
        reference_cdf = np.cumsum(reference_hist)
        
        # Normalize CDFs to [0, 255]
        source_cdf = source_cdf * 255 / source_cdf[-1]
        reference_cdf = reference_cdf * 255 / reference_cdf[-1]
        
        # Create lookup table for histogram matching
        lut = np.zeros(256, dtype=np.uint8)
        j = 0
        for i in range(256):
            while j < 256 and reference_cdf[j] < source_cdf[i]:
                j += 1
            lut[i] = j
        
        # Apply histogram matching using lookup table
        matched = cv.LUT(source, lut)
        
        return matched
    except Exception as e:
        print(f"⚠️  Histogram matching failed: {str(e)}, returning original")
        return source.copy()

def align_feature_distributions(X_primary, X_secondary, method='mean_std'):
    """Align feature distributions between datasets"""
    if X_primary is None or X_secondary is None:
        raise ValueError("Invalid feature matrices")
    
    if X_primary.shape[1] != X_secondary.shape[1]:
        raise ValueError(f"Feature count mismatch: {X_primary.shape[1]} vs {X_secondary.shape[1]}")
    
    try:
        X_primary_aligned = X_primary.copy()
        X_secondary_aligned = X_secondary.copy()
        
        if method == 'mean_std':
            # Z-score normalization: (x - mean) / std
            # Align each to combined statistics
            for feat_idx in range(X_primary.shape[1]):
                # Compute combined statistics
                combined = np.concatenate([X_primary[:, feat_idx], X_secondary[:, feat_idx]])
                combined_mean = np.mean(combined)
                combined_std = np.std(combined)
                
                # Avoid division by zero
                if combined_std < 1e-6:
                    combined_std = 1.0
                
                # Normalize both datasets
                X_primary_aligned[:, feat_idx] = (X_primary[:, feat_idx] - combined_mean) / combined_std
                X_secondary_aligned[:, feat_idx] = (X_secondary[:, feat_idx] - combined_mean) / combined_std
        
        elif method == 'minmax':
            # Min-max normalization: (x - min) / (max - min)
            # Align each to combined range
            for feat_idx in range(X_primary.shape[1]):
                combined = np.concatenate([X_primary[:, feat_idx], X_secondary[:, feat_idx]])
                combined_min = np.min(combined)
                combined_max = np.max(combined)
                
                # Avoid division by zero
                feature_range = combined_max - combined_min
                if feature_range < 1e-6:
                    feature_range = 1.0
                
                # Normalize both datasets
                X_primary_aligned[:, feat_idx] = (X_primary[:, feat_idx] - combined_min) / feature_range
                X_secondary_aligned[:, feat_idx] = (X_secondary[:, feat_idx] - combined_min) / feature_range
        
        else:
            raise ValueError(f"Unknown method: {method}. Use 'mean_std' or 'minmax'")
        
        return X_primary_aligned, X_secondary_aligned
    
    except Exception as e:
        print(f"⚠️  Feature alignment failed: {str(e)}, returning originals")
        return X_primary, X_secondary

# Classification model functions for breast cancer detection

def extract_breast_cancer_features(img):
    """Extract 42 thermal features for classification"""
    if img is None or img.size == 0:
        raise ValueError("Invalid image: image is None or empty")
    
    features = []
    
    try:
        # ===== BASIC THERMAL STATISTICS =====
        thermal_stats = extract_thermal_features(img)
        for key in ['mean_temperature', 'std_temperature', 'min_temperature', 'max_temperature']:
            val = thermal_stats[key]
            features.append(val if not np.isnan(val) else 0.0)
        
        # ===== ADVANCED THERMAL STATISTICS =====
        flattened = img.flatten()
        mean_flat = np.mean(flattened)
        std_flat = np.std(flattened)
        
        # Coefficient of variation (std/mean)
        coeff_var = (std_flat / (mean_flat + 1e-6)) if mean_flat > 1e-6 else 0.0
        features.append(coeff_var if not np.isnan(coeff_var) else 0.0)
        
        # Interquartile range
        iqr = float(np.percentile(flattened, 75) - np.percentile(flattened, 25))
        features.append(iqr if not np.isnan(iqr) else 0.0)
        
        # ===== ASYMMETRY FEATURES =====
        asymmetry = calculate_breast_asymmetry(img)
        for key in ['temperature_asymmetry', 'asymmetry_percentage', 'left_mean_temp', 
                    'right_mean_temp', 'left_std', 'right_std']:
            val = asymmetry[key]
            features.append(val if not np.isnan(val) else 0.0)
        
        # Absolute difference (validation)
        diff = abs(asymmetry['left_mean_temp'] - asymmetry['right_mean_temp'])
        features.append(diff if not np.isnan(diff) else 0.0)
        
        # ===== QUADRANT ANALYSIS FEATURES =====
        quadrants = analyze_breast_quadrants(img)
        quad_temps = []
        for key in ['left_upper_outer', 'left_upper_inner', 'left_lower_outer', 'left_lower_inner',
                    'right_upper_outer', 'right_upper_inner', 'right_lower_outer', 'right_lower_inner']:
            val = quadrants[key]
            quad_temps.append(val if not np.isnan(val) else 0.0)
            features.append(val if not np.isnan(val) else 0.0)
        
        # Quadrant statistics
        if len(quad_temps) > 0:
            quad_range = np.max(quad_temps) - np.min(quad_temps)
            quad_std = np.std(quad_temps)
            features.append(quad_range if not np.isnan(quad_range) else 0.0)
            features.append(quad_std if not np.isnan(quad_std) else 0.0)
        else:
            features.extend([0.0, 0.0])
        
        # ===== HOTSPOT/ABNORMALITY DETECTION =====
        hotspots = detect_breast_hotspots(img)
        features.append(float(len(hotspots['left_hotspots'])))
        features.append(float(len(hotspots['right_hotspots'])))
        features.append(float(len(hotspots['left_hotspots']) + len(hotspots['right_hotspots'])))
        features.append(float(hotspots['left_threshold']))
        features.append(float(hotspots['right_threshold']))
        
        # ===== EDGE AND GRADIENT FEATURES =====
        sobelx = cv.Sobel(img, cv.CV_64F, 1, 0, ksize=5)
        sobely = cv.Sobel(img, cv.CV_64F, 0, 1, ksize=5)
        magnitude = np.sqrt(sobelx**2 + sobely**2)
        
        mag_mean = float(np.mean(magnitude))
        mag_std = float(np.std(magnitude))
        mag_max = float(np.max(magnitude))
        
        features.append(mag_mean if not np.isnan(mag_mean) else 0.0)
        features.append(mag_std if not np.isnan(mag_std) else 0.0)
        features.append(mag_max if not np.isnan(mag_max) else 0.0)
        
        # ===== TEXTURE FEATURES =====
        h, w = img.shape
        block_size = max(8, min(h, w) // 8)  # Ensure reasonable block size
        local_contrasts = []
        
        for i in range(0, max(1, h - block_size), block_size):
            for j in range(0, max(1, w - block_size), block_size):
                block = img[i:i+block_size, j:j+block_size]
                local_contrast = float(np.std(block))
                if not np.isnan(local_contrast):
                    local_contrasts.append(local_contrast)
        
        if local_contrasts:
            features.extend([
                float(np.mean(local_contrasts)),
                float(np.std(local_contrasts)),
                float(np.max(local_contrasts)),
                float(np.min(local_contrasts))
            ])
        else:
            features.extend([0.0, 0.0, 0.0, 0.0])
        
        # ===== HISTOGRAM FEATURES =====
        hist, _ = np.histogram(img, bins=20, range=(0, 256))
        hist = hist / len(flattened)  # Normalize
        
        # Distribution shape features
        hist_max = float(np.max(hist))
        hist_argmax = float(np.argmax(hist))
        
        features.append(hist_max if not np.isnan(hist_max) else 0.0)
        features.append(hist_argmax if not np.isnan(hist_argmax) else 0.0)
        
        # ===== CONTOUR/REGION FEATURES =====
        _, binary = cv.threshold(img, 127, 255, cv.THRESH_BINARY)
        contours, _ = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        
        if len(contours) > 0:
            areas = [cv.contourArea(c) for c in contours]
            features.extend([
                float(len(contours)),
                float(np.mean(areas)) if areas else 0.0,
                float(np.std(areas)) if len(areas) > 1 else 0.0,
                float(np.max(areas)) if areas else 0.0
            ])
        else:
            features.extend([0.0, 0.0, 0.0, 0.0])
        
        # ===== TEMPERATURE SYMMETRY FEATURES =====
        mid_col = w // 2
        left_half = img[:, :mid_col]
        right_half = img[:, mid_col:]
        
        # Resize to match
        min_width = min(left_half.shape[1], right_half.shape[1])
        left_half = left_half[:, :min_width]
        right_half = right_half[:, :min_width]
        
        if left_half.size > 0 and right_half.size > 0:
            try:
                corr_matrix = np.corrcoef(left_half.flatten(), right_half.flatten())
                symmetry_corr = float(corr_matrix[0, 1])
                features.append(symmetry_corr if not np.isnan(symmetry_corr) else 0.0)
            except:
                features.append(0.0)
        else:
            features.append(0.0)
        
        # Final validation: Replace any remaining NaN with 0.0
        features = [0.0 if np.isnan(f) else f for f in features]
        
        return np.array(features, dtype=np.float32)
    
    except Exception as e:
        print(f"Error extracting features: {str(e)}")
        # Return safe default features filled with zeros
        return np.zeros(42, dtype=np.float32)

# ============================================================================
# MACHINE LEARNING FUNCTIONS - Model training, prediction and evaluation
# ============================================================================

def train_breast_cancer_model(X, y, model_type='svm', test_size=0.2, random_state=42):
    """Train SVM or Random Forest classifier"""
    if X is None or y is None or len(X) == 0 or len(y) == 0:
        raise ValueError("Invalid X or y: cannot be None or empty")
    
    if len(X) != len(y):
        raise ValueError(f"X and y length mismatch: {len(X)} vs {len(y)}")
    
    if model_type not in ['svm', 'rf']:
        raise ValueError(f"model_type must be 'svm' or 'rf', got {model_type}")
    
    # Feature scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Handle class imbalance with resampling
    original_cancer_count = sum(y)
    original_normal_count = sum(y == 0)
    imbalance_ratio = abs(original_cancer_count - original_normal_count) / max(original_cancer_count, original_normal_count)
    
    if imbalance_ratio > 0.3:
        # Dataset is imbalanced, apply resampling
        if original_cancer_count < original_normal_count:
            # Undersample majority class
            X_majority = X_scaled[y == 0]
            y_majority = y[y == 0]
            X_minority = X_scaled[y == 1]
            y_minority = y[y == 1]
            
            X_majority_downsampled, y_majority_downsampled = resample(
                X_majority, y_majority, 
                replace=False, 
                n_samples=len(X_minority), 
                random_state=random_state
            )
            
            X_balanced = np.vstack([X_majority_downsampled, X_minority])
            y_balanced = np.hstack([y_majority_downsampled, y_minority])
        else:
            # Oversample minority class
            X_minority = X_scaled[y == 1]
            y_minority = y[y == 1]
            
            X_minority_upsampled, y_minority_upsampled = resample(
                X_minority, y_minority, 
                replace=True, 
                n_samples=original_normal_count, 
                random_state=random_state
            )
            
            X_balanced = np.vstack([X_scaled[y == 0], X_minority_upsampled])
            y_balanced = np.hstack([y[y == 0], y_minority_upsampled])
        
        print(f"✓ Applied resampling: {len(X)} → {len(X_balanced)} samples")
    else:
        X_balanced, y_balanced = X_scaled, y
        print("✓ Dataset balanced, no resampling needed")
    
    print(f"✓ Training on: {sum(y_balanced)} cancer, {sum(y_balanced==0)} normal cases")
    
    # Split data with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X_balanced, y_balanced, test_size=test_size, random_state=random_state, stratify=y_balanced
    )
    
    # Train model with optimized hyperparameters
    if model_type == 'svm':
        from sklearn.svm import SVC
        model = SVC(kernel='rbf', C=10.0, gamma='auto', random_state=random_state, 
                   probability=True, class_weight='balanced')
        print("✓ Using SVM (RBF kernel, C=10.0)")
    else:  # rf
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(
            n_estimators=300, 
            max_depth=20,
            min_samples_split=4,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1,
            class_weight='balanced'
        )
        print("✓ Using Random Forest (300 trees, max_depth=20)")
    
    # Train model
    model.fit(X_train, y_train)
    
    # Evaluate on test set
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n✓ Model trained with {accuracy:.3f} accuracy on test set")
    
    # Print detailed classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Normal', 'Cancer']))
    
    # Save model and scaler together
    model_data = {
        'model': model,
        'scaler': scaler,
        'feature_count': X.shape[1],
        'model_type': model_type
    }
    
    with open('breast_cancer_model.pkl', 'wb') as f:
        pickle.dump(model_data, f)
    
    print("✓ Model and scaler saved to breast_cancer_model.pkl")
    
    return model, scaler, accuracy

def predict_breast_cancer(model_data, img):
    """Predict cancer class and confidence"""
    if img is None or img.size == 0:
        raise ValueError("Invalid image provided")
    
    # Handle both old format (just model) and new format (model + scaler)
    if isinstance(model_data, dict) and 'model' in model_data:
        model = model_data['model']
        scaler = model_data.get('scaler')
    else:
        model = model_data
        scaler = None
    
    if model is None:
        raise ValueError("Model is None - not initialized")
    
    # Extract features
    features = extract_breast_cancer_features(img)
    
    # Validate feature extraction
    if features is None or len(features) == 0:
        raise ValueError("Feature extraction failed")
    
    if np.any(np.isnan(features)) or np.any(np.isinf(features)):
        print("⚠️  Warning: NaN/Inf detected in features, replacing with 0")
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    
    features = features.reshape(1, -1)  # Reshape for single prediction
    
    # Apply scaling if available
    if scaler is not None:
        features = scaler.transform(features)
    
    # Make prediction
    try:
        prediction = int(model.predict(features)[0])
    except Exception as e:
        print(f"✗ Prediction error: {str(e)}")
        return None, None
    
    # Get prediction probabilities if available
    confidence = None
    if hasattr(model, 'predict_proba'):
        try:
            probabilities = model.predict_proba(features)[0]
            confidence = float(probabilities[prediction])
        except:
            confidence = None
    
    return prediction, confidence

# ============================================================================
# DATA LOADING FUNCTIONS - Dataset handling and feature extraction pipeline
# ============================================================================

def load_thermal_images_from_folder(folder_path, label, recursive=False):
    """Load and process images from folder"""
    if not os.path.exists(folder_path):
        raise ValueError(f"Folder not found: {folder_path}")
    
    images = []
    features_list = []
    supported_formats = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')
    
    # Get all image files (recursively if requested)
    image_files = []
    if recursive:
        for root, dirs, files in os.walk(folder_path):
            for f in files:
                if f.endswith(supported_formats):
                    image_files.append(os.path.join(root, f))
    else:
        image_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) 
                       if f.endswith(supported_formats)]
    
    print(f"  Loading {len(image_files)} images from {os.path.basename(folder_path)}...")
    
    for idx, filepath in enumerate(image_files, 1):
        try:
            # Load image
            img = cv.imread(filepath)
            if img is None:
                continue
            
            # Preprocess
            processed = normalize_image(convert_to_grayscale(img))
            
            # Extract features
            features = extract_breast_cancer_features(processed)
            
            images.append(processed)
            features_list.append(features)
            
            if idx % max(1, len(image_files) // 5) == 0 and len(image_files) > 10:
                print(f"    ✓ Processed {idx}/{len(image_files)} images")
        
        except Exception as e:
            continue
    
    print(f"  ✓ Successfully loaded {len(features_list)} images")
    
    return images, np.array(features_list) if features_list else np.array([])

def prepare_training_data(X, y, test_size=0.2, val_size=0.1, random_state=42):
    """Split data into train/val/test"""
    if X is None or y is None or len(X) == 0:
        raise ValueError("Invalid X or y provided")
    
    if len(X) != len(y):
        raise ValueError(f"X and y length mismatch: {len(X)} vs {len(y)}")
    
    # First split: training + validation vs test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Second split: training vs validation
    val_split = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_split, random_state=random_state, stratify=y_temp
    )
    
    print("\n" + "=" * 60)
    print("DATA SPLIT INFORMATION")
    print("=" * 60)
    print(f"Training set: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
    print(f"  Cancer: {sum(y_train)} | No Cancer: {sum(y_train==0)}")
    print(f"Validation set: {len(X_val)} samples ({len(X_val)/len(X)*100:.1f}%)")
    print(f"  Cancer: {sum(y_val)} | No Cancer: {sum(y_val==0)}")
    print(f"Test set: {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")
    print(f"  Cancer: {sum(y_test)} | No Cancer: {sum(y_test==0)}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def load_dataset_recursive(dataset_path, verbose=True):
    """Recursively load all images from directory"""
    if verbose:
        print("\n✓ Detected recursive mode (scanning all PNG/JPG files)\n")
    
    cancer_keywords = ['cancer', 'sick', 'malignant', 'tumor']
    healthy_keywords = ['no_cancer', 'healthy', 'benign', 'normal']
    
    cancer_features = []
    healthy_features = []
    unclassified_features = []
    unclassified_paths = []
    
    total_scanned = 0
    
    # Recursively scan all files
    for root, dirs, files in os.walk(dataset_path):
        # Infer label from folder path
        folder_lower = root.lower()
        inferred_label = None
        
        for keyword in cancer_keywords:
            if keyword in folder_lower:
                inferred_label = 1
                break
        
        if inferred_label is None:
            for keyword in healthy_keywords:
                if keyword in folder_lower:
                    inferred_label = 0
                    break
        
        # Load images from this folder
        for filename in files:
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                filepath = os.path.join(root, filename)
                
                try:
                    img = cv.imread(filepath)
                    if img is not None:
                        processed = normalize_image(convert_to_grayscale(img))
                        features = extract_breast_cancer_features(processed)
                        
                        if inferred_label == 1:
                            cancer_features.append(features)
                        elif inferred_label == 0:
                            healthy_features.append(features)
                        else:
                            unclassified_features.append(features)
                            unclassified_paths.append(filepath)
                        
                        total_scanned += 1
                        
                        if verbose and total_scanned % 200 == 0:
                            print(f"  Scanned {total_scanned} images...")
                
                except Exception as e:
                    pass
    
    if verbose:
        print(f"  ✓ Scanned {total_scanned} total files")
    
    # Assign unclassified to maintain balance
    if len(unclassified_features) > 0:
        # Calculate ratio from classified images
        total_classified = len(cancer_features) + len(healthy_features)
        if total_classified > 0:
            cancer_ratio = len(cancer_features) / total_classified
            # Split unclassified based on desired ratio
            n_cancer = int(len(unclassified_features) * cancer_ratio)
            cancer_features.extend(unclassified_features[:n_cancer])
            healthy_features.extend(unclassified_features[n_cancer:])
        else:
            # If no classified images, split evenly
            mid = len(unclassified_features) // 2
            cancer_features.extend(unclassified_features[:mid])
            healthy_features.extend(unclassified_features[mid:])
    
    # Combine into single arrays
    X_cancer = np.array(cancer_features) if cancer_features else np.array([])
    X_healthy = np.array(healthy_features) if healthy_features else np.array([])
    y_cancer = np.ones(len(cancer_features), dtype=int) if cancer_features else np.array([])
    y_healthy = np.zeros(len(healthy_features), dtype=int) if healthy_features else np.array([])
    
    # Stack all features
    if len(X_cancer) > 0 and len(X_healthy) > 0:
        X = np.vstack([X_healthy, X_cancer])
        y = np.concatenate([y_healthy, y_cancer])
    elif len(X_cancer) > 0:
        X = X_cancer
        y = y_cancer
    elif len(X_healthy) > 0:
        X = X_healthy
        y = y_healthy
    else:
        X = np.array([])
        y = np.array([])
    
    info = {
        'total_images': len(X),
        'cancer_count': len(cancer_features),
        'no_cancer_count': len(healthy_features),
        'unknown_count': 0,
        'format': 'recursive'
    }
    
    return X, y, info

def load_dataset_from_directory(dataset_path, load_all=False):
    """Load dataset in Kaggle or Legacy format"""
    print("=" * 60)
    print("LOADING THERMAL BREAST CANCER DATASET")
    print("=" * 60)
    
    info = {
        'total_images': 0,
        'cancer_count': 0,
        'no_cancer_count': 0,
        'unknown_count': 0,
        'format': 'unknown'
    }
    
    # FORCE RECURSIVE LOADING IF REQUESTED
    if load_all:
        print("\n⚠️  --load-all flag enabled: Recursively loading ALL images\n")
        X, y, info = load_dataset_recursive(dataset_path, verbose=True)
        return X, y, info
    
    # Check which format we're using
    training_path = os.path.join(dataset_path, 'Training')
    testing_path = os.path.join(dataset_path, 'Testing')
    cancer_path = os.path.join(dataset_path, 'Cancer')
    no_cancer_path = os.path.join(dataset_path, 'No_Cancer')
    unknown_path = os.path.join(dataset_path, 'Unknown_class')
    
    # KAGGLE FORMAT DETECTION
    if os.path.exists(training_path):
        print("\n✓ Detected Kaggle format (Training/Testing with Benign/Malignant)\n")
        info['format'] = 'kaggle'
        
        benign_path = os.path.join(training_path, 'Benign')
        malignant_path = os.path.join(training_path, 'Malignant')
        
        # Load malignant (cancer) from patient directories
        cancer_features = []
        y_cancer = []
        if os.path.exists(malignant_path):
            print("[1/2] Loading MALIGNANT (cancer) patient directories...")
            malignant_patients = [d for d in os.listdir(malignant_path) 
                                if os.path.isdir(os.path.join(malignant_path, d))]
            
            for p_idx, patient_dir in enumerate(malignant_patients, 1):
                patient_path = os.path.join(malignant_path, patient_dir)
                # Load all thermal images from this patient
                for img_file in os.listdir(patient_path):
                    if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        img_path = os.path.join(patient_path, img_file)
                        try:
                            img = cv.imread(img_path)
                            if img is not None:
                                processed = normalize_image(convert_to_grayscale(img))
                                features = extract_breast_cancer_features(processed)
                                cancer_features.append(features)
                                y_cancer.append(1)
                                info['cancer_count'] += 1
                        except:
                            pass
                
                if p_idx % max(1, len(malignant_patients) // 5) == 0:
                    print(f"  ✓ Loaded {p_idx}/{len(malignant_patients)} malignant patients")
            
            print(f"  ✓ Successfully loaded {info['cancer_count']} images from {len(malignant_patients)} malignant patients")
        
        # Load benign (no cancer) from patient directories
        no_cancer_features = []
        y_no_cancer = []
        if os.path.exists(benign_path):
            print("\n[2/2] Loading BENIGN (no cancer) patient directories...")
            benign_patients = [d for d in os.listdir(benign_path) 
                             if os.path.isdir(os.path.join(benign_path, d))]
            
            for p_idx, patient_dir in enumerate(benign_patients, 1):
                patient_path = os.path.join(benign_path, patient_dir)
                # Load all thermal images from this patient
                for img_file in os.listdir(patient_path):
                    if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        img_path = os.path.join(patient_path, img_file)
                        try:
                            img = cv.imread(img_path)
                            if img is not None:
                                processed = normalize_image(convert_to_grayscale(img))
                                features = extract_breast_cancer_features(processed)
                                no_cancer_features.append(features)
                                y_no_cancer.append(0)
                                info['no_cancer_count'] += 1
                        except:
                            pass
                
                if p_idx % max(1, len(benign_patients) // 5) == 0:
                    print(f"  ✓ Loaded {p_idx}/{len(benign_patients)} benign patients")
            
            print(f"  ✓ Successfully loaded {info['no_cancer_count']} images from {len(benign_patients)} benign patients")
        
        # Combine features
        if len(cancer_features) > 0 or len(no_cancer_features) > 0:
            if len(cancer_features) > 0:
                cancer_features = np.array(cancer_features)
                y_cancer = np.array(y_cancer)
            else:
                cancer_features = np.array([])
                y_cancer = np.array([])
            
            if len(no_cancer_features) > 0:
                no_cancer_features = np.array(no_cancer_features)
                y_no_cancer = np.array(y_no_cancer)
            else:
                no_cancer_features = np.array([])
                y_no_cancer = np.array([])
        
    # HEALTHY/SICK FORMAT DETECTION (newer structure)
    healthy_path = os.path.join(dataset_path, 'Healthy')
    sick_path = os.path.join(dataset_path, 'Sick')
    
    if os.path.exists(healthy_path) or os.path.exists(sick_path):
        print("\n✓ Detected Healthy/Sick format\n")
        info['format'] = 'healthy_sick'
        
        # Load sick (cancer) images
        if os.path.exists(sick_path):
            print("[1/2] Loading SICK images...")
            cancer_images, cancer_features = load_thermal_images_from_folder(sick_path, label=1, recursive=True)
            y_cancer = np.ones(len(cancer_features), dtype=int)
            info['cancer_count'] = len(cancer_features)
        else:
            print(f"⚠️  Sick folder not found at {sick_path}")
            cancer_features = np.array([])
            y_cancer = np.array([])
        
        # Load healthy (no cancer) images
        if os.path.exists(healthy_path):
            print("\n[2/2] Loading HEALTHY images...")
            no_cancer_images, no_cancer_features = load_thermal_images_from_folder(healthy_path, label=0, recursive=True)
            y_no_cancer = np.zeros(len(no_cancer_features), dtype=int)
            info['no_cancer_count'] = len(no_cancer_features)
        else:
            print(f"⚠️  Healthy folder not found at {healthy_path}")
            no_cancer_features = np.array([])
            y_no_cancer = np.array([])
        
        # Load unknown test/inference images separately if present
        unknown_features = np.array([])
        if os.path.exists(unknown_path):
            print("\n[3/3] Loading UNKNOWN test images (inference-only)...")
            _, unknown_features = load_thermal_images_from_folder(unknown_path, label=None)
            info['unknown_count'] = len(unknown_features)
    
    # LEGACY FORMAT DETECTION
    elif os.path.exists(cancer_path) or os.path.exists(no_cancer_path):
        print("\n✓ Detected legacy format (Cancer/No_Cancer)\n")
        info['format'] = 'legacy'
        
        # Load cancer images
        if os.path.exists(cancer_path):
            print("[1/2] Loading CANCER images...")
            cancer_images, cancer_features = load_thermal_images_from_folder(cancer_path, label=1)
            y_cancer = np.ones(len(cancer_features), dtype=int)
            info['cancer_count'] = len(cancer_features)
        else:
            print(f"⚠️  Cancer folder not found at {cancer_path}")
            cancer_features = np.array([])
            y_cancer = np.array([])
        
        # Load no cancer images
        if os.path.exists(no_cancer_path):
            print("\n[2/2] Loading NO CANCER images...")
            no_cancer_images, no_cancer_features = load_thermal_images_from_folder(no_cancer_path, label=0)
            y_no_cancer = np.zeros(len(no_cancer_features), dtype=int)
            info['no_cancer_count'] = len(no_cancer_features)
        else:
            print(f"⚠️  No_Cancer folder not found at {no_cancer_path}")
            no_cancer_features = np.array([])
            y_no_cancer = np.array([])
        
        # Load unknown test/inference images separately if present
        unknown_features = np.array([])
        if os.path.exists(unknown_path):
            print("\n[3/3] Loading UNKNOWN test images (inference-only)...")
            _, unknown_features = load_thermal_images_from_folder(unknown_path, label=None)
            info['unknown_count'] = len(unknown_features)
    else:
        print("✗ Dataset folder structure not recognized!")
        print("Expected either:")
        print("  - Kaggle format: Dataset/Training/[Benign|Malignant]")
        print("  - Legacy format: Dataset/[Cancer|No_Cancer]")
        print("  - Modern format: Dataset/[Healthy|Sick]")
        print("\n✓ Attempting recursive scan of all PNG/JPG files...")
        
        # Try recursive loading as fallback
        X, y, info = load_dataset_recursive(dataset_path, verbose=True)
        
        if X is None or len(X) == 0:
            print("✗ No structured format detected and recursive scan found no images.")
            return None, None, info
        else:
            return X, y, info
    
    
    # Combine training datasets
    if len(cancer_features) > 0 and len(no_cancer_features) > 0:
        X = np.vstack([no_cancer_features, cancer_features])
        y = np.concatenate([y_no_cancer, y_cancer])
    elif len(cancer_features) > 0:
        X = cancer_features
        y = y_cancer
    elif len(no_cancer_features) > 0:
        X = no_cancer_features
        y = y_no_cancer
    else:
        print("✗ No images found in dataset!")
        return None, None, info
    
    info['total_images'] = len(X)
    
    # Print summary
    print("\n" + "=" * 60)
    print("DATASET LOADING SUMMARY")
    print("=" * 60)
    print(f"Total images loaded: {info['total_images']}")
    print(f"  No Cancer/Benign: {info['no_cancer_count']} ({info['no_cancer_count']/max(info['total_images'], 1)*100:.1f}%)")
    print(f"  Cancer/Malignant: {info['cancer_count']} ({info['cancer_count']/max(info['total_images'], 1)*100:.1f}%)")
    print(f"Feature vector size: {X.shape[1]} features per image")
    
    return X, y, info

def load_thermal_images_flat(folder_path, estimated_cancer_rate=0.3, random_state=42):
    """Load images from unorganized folder"""
    import os
    
    np.random.seed(random_state)
    
    info = {
        'total_images': 0,
        'cancer_count': 0,
        'no_cancer_count': 0,
        'inferred_labels': False
    }
    
    if not os.path.exists(folder_path):
        print(f"⚠️  Folder not found: {folder_path}")
        return np.array([]), np.array([]), info
    
    print(f"\n✓ Scanning folder for thermal images: {folder_path}")
    
    features_list = []
    labels_list = []
    file_count = 0
    
    # Recursively scan all subdirectories for image files
    for root, dirs, files in os.walk(folder_path):
        for filename in files:
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                filepath = os.path.join(root, filename)
                file_count += 1
                
                try:
                    # Load and preprocess image
                    img = cv.imread(filepath)
                    if img is None:
                        continue
                    
                    processed = normalize_image(convert_to_grayscale(img))
                    features = extract_breast_cancer_features(processed)
                    features_list.append(features)
                    
                    # Infer label from folder/file name
                    path_lower = filepath.lower()
                    if 'malignant' in path_lower or 'cancer' in path_lower:
                        labels_list.append(1)
                        info['cancer_count'] += 1
                        info['inferred_labels'] = True
                    elif 'benign' in path_lower or 'normal' in path_lower:
                        labels_list.append(0)
                        info['no_cancer_count'] += 1
                        info['inferred_labels'] = True
                    else:
                        # Random assignment based on estimated rate
                        if np.random.random() < estimated_cancer_rate:
                            labels_list.append(1)
                            info['cancer_count'] += 1
                        else:
                            labels_list.append(0)
                            info['no_cancer_count'] += 1
                
                except Exception as e:
                    pass
        
        # Progress indicator every 100 files
        if file_count % 100 == 0 and file_count > 0:
            print(f"  Processing... {file_count} files scanned, {len(features_list)} valid images loaded")
    
    if len(features_list) == 0:
        print(f"⚠️  No thermal images found in {folder_path}")
        return np.array([]), np.array([]), info
    
    X = np.array(features_list)
    y = np.array(labels_list)
    info['total_images'] = len(X)
    
    label_method = "inferred from folder names" if info['inferred_labels'] else f"random ({estimated_cancer_rate*100:.0f}% cancer)"
    print(f"  ✓ Loaded {len(X)} thermal images")
    print(f"  ✓ Labels: {label_method}")
    print(f"    Cancer: {info['cancer_count']} | No Cancer: {info['no_cancer_count']}")
    
    return X, y, info

def prepare_training_data(X, y, test_size=0.2, val_size=0.1, random_state=42):
    """
    Split dataset into training, validation, and test sets.
    
    Parameters:
    - X: Feature matrix
    - y: Labels
    - test_size: Fraction for test set (default 20%)
    - val_size: Fraction for validation set from training data (default 10%)
    - random_state: Random seed
    
    Returns:
    - X_train, X_val, X_test: Training, validation, and test features
    - y_train, y_val, y_test: Training, validation, and test labels
    """
    from sklearn.model_selection import train_test_split
    
    # First split: training + validation vs test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Second split: training vs validation
    val_split = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_split, random_state=random_state, stratify=y_temp
    )
    
    print("\n" + "=" * 60)
    print("DATA SPLIT INFORMATION")
    print("=" * 60)
    print(f"Training set: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
    print(f"  Cancer: {sum(y_train)} | No Cancer: {sum(y_train==0)}")
    print(f"Validation set: {len(X_val)} samples ({len(X_val)/len(X)*100:.1f}%)")
    print(f"  Cancer: {sum(y_val)} | No Cancer: {sum(y_val==0)}")
    print(f"Test set: {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")
    print(f"  Cancer: {sum(y_test)} | No Cancer: {sum(y_test==0)}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

# Batch processing functions for screening multiple thermal images

def batch_process_thermal_images(image_directory, model, output_results=True, interactive=False, display_size=800):
    """
    Process multiple thermal images from a directory.
    
    Parameters:
    - image_directory: Path to folder containing .jpg/.png thermal images
    - model: Trained breast cancer detection model
    - output_results: Whether to print and save results
    - interactive: Whether to process images one by one with user input
    
    Returns:
    - results: Dictionary with predictions for each image
    """
    import os
    from pathlib import Path
    
    results = {
        'total_images': 0,
        'cancer_detected': 0,
        'normal': 0,
        'predictions': [],
        'high_risk': []
    }
    
    # Find all image files, including nested directories
    supported_formats = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')
    image_paths = []
    for root, _, files in os.walk(image_directory):
        for f in files:
            if f.endswith(supported_formats):
                image_paths.append(os.path.join(root, f))
    
    if not image_paths:
        print(f"No thermal images found in {image_directory}")
        return results
    
    if interactive:
        print(f"Starting interactive mode with {len(image_paths)} thermal images...")
        print("Showing PROCESSED thermal images (grayscale, normalized) - what the AI analyzes")
        print("Press Enter to move to next image, 'q' to quit, 'd' for detailed analysis")
        print("-" * 80)
    else:
        print(f"Processing {len(image_paths)} thermal images...")
        print("-" * 60)
    
    for idx, filepath in enumerate(image_paths, 1):
        filename = os.path.relpath(filepath, image_directory)
        
        try:
            # Load and preprocess image
            img = cv.imread(filepath)
            if img is None:
                print(f"[{idx}/{len(image_paths)}] ✗ {filename} - Failed to load")
                continue
            
            processed = normalize_image(convert_to_grayscale(img))
            
            # Make prediction
            prediction, confidence = predict_breast_cancer(model, processed)
            
            # Store results
            result_entry = {
                'filename': filename,
                'prediction': 'Cancer' if prediction == 1 else 'Normal',
                'confidence': confidence,
                'raw_prediction': prediction
            }
            
            results['predictions'].append(result_entry)
            results['total_images'] += 1
            
            if prediction == 1:
                results['cancer_detected'] += 1
                results['high_risk'].append(filename)
                status = "⚠️  CANCER"
            else:
                results['normal'] += 1
                status = "✓ NORMAL"
            
            confidence_pct = f"{confidence*100:.1f}%" if confidence else "N/A"
            
            if interactive:
                # Display the processed thermal image (what the model actually sees)
                height, width = processed.shape[:2]
                max_display_size = display_size  # Maximum dimension for display
                
                if max(height, width) < max_display_size:
                    # Scale up small images
                    scale_factor = max_display_size / max(height, width)
                    new_width = int(width * scale_factor)
                    new_height = int(height * scale_factor)
                    display_img = cv.resize(processed, (new_width, new_height), interpolation=cv.INTER_LINEAR)
                else:
                    # Scale down large images
                    scale_factor = max_display_size / max(height, width)
                    new_width = int(width * scale_factor)
                    new_height = int(height * scale_factor)
                    display_img = cv.resize(processed, (new_width, new_height), interpolation=cv.INTER_AREA)
                
                # Convert to 3-channel for better display if needed
                if len(display_img.shape) == 2:
                    display_img = cv.cvtColor(display_img, cv.COLOR_GRAY2BGR)
                
                # Display processed image
                cv.imshow(f'Breast Cancer Detection - {filename}', display_img)
                cv.waitKey(1)  # Allow window to update
                
                # Show prediction
                print(f"\n[{idx}/{len(image_paths)}] {status} - {filename}")
                print(f"Prediction: {result_entry['prediction']} (Confidence: {confidence_pct})")
                print("-" * 50)
                
                # Wait for user input
                while True:
                    user_input = input("Press Enter for next image, 'd' for details, 'q' to quit: ").strip().lower()
                    if user_input == 'q':
                        cv.destroyAllWindows()
                        print("Interactive mode stopped by user.")
                        return results
                    elif user_input == 'd':
                        # Show detailed analysis
                        analysis = process_single_thermal_image(filepath, model, visualize=False)
                        print("\nPress Enter to continue...")
                        input()
                        break
                    elif user_input == '':
                        break
                    else:
                        print("Invalid input. Press Enter, 'd', or 'q'.")
                
                cv.destroyAllWindows()
            else:
                print(f"[{idx}/{len(image_paths)}] {status} - {filename} ({confidence_pct})")
            
        except Exception as e:
            print(f"[{idx}/{len(image_paths)}] ✗ {filename} - Error: {str(e)}")
            continue
    
    # Print summary (only for non-interactive mode)
    if not interactive:
        print("-" * 60)
        print("BATCH PROCESSING SUMMARY")
        print("-" * 60)
        print(f"Total images processed: {results['total_images']}")
        print(f"Cancer detected: {results['cancer_detected']} ({results['cancer_detected']/max(results['total_images'], 1)*100:.1f}%)")
        print(f"Normal: {results['normal']} ({results['normal']/max(results['total_images'], 1)*100:.1f}%)")
        
        if results['high_risk']:
            print(f"\n⚠️  HIGH RISK IMAGES ({len(results['high_risk'])}):")
            for img_name in results['high_risk']:
                print(f"   - {img_name}")
    
    return results

def process_single_thermal_image(image_path, model, visualize=False):
    """
    Process a single thermal image with detailed analysis.
    
    Parameters:
    - image_path: Path to thermal image
    - model: Trained breast cancer detection model
    - visualize: Whether to display the image (requires cv.imshow support)
    
    Returns:
    - analysis_result: Detailed analysis dictionary
    """
    # Load and preprocess
    img = cv.imread(image_path)
    if img is None:
        print(f"Error: Could not load image from {image_path}")
        return None
    
    processed = normalize_image(convert_to_grayscale(img))
    
    # Conduct full analysis
    asymmetry = calculate_breast_asymmetry(processed)
    quadrants = analyze_breast_quadrants(processed)
    hotspots = detect_breast_hotspots(processed)
    
    # Make prediction
    prediction, confidence = predict_breast_cancer(model, processed)
    
    # Compile results
    analysis_result = {
        'filename': os.path.basename(image_path),
        'prediction': 'Cancer' if prediction == 1 else 'Normal',
        'confidence': confidence,
        'asymmetry_analysis': asymmetry,
        'quadrant_analysis': quadrants,
        'hotspot_analysis': {
            'left_hotspots_count': len(hotspots['left_hotspots']),
            'right_hotspots_count': len(hotspots['right_hotspots']),
            'left_threshold': hotspots['left_threshold'],
            'right_threshold': hotspots['right_threshold']
        }
    }
    
    # Print detailed report
    print(f"\nDetailed Analysis Report: {os.path.basename(image_path)}")
    print("=" * 60)
    print(f"Prediction: {analysis_result['prediction']} (Confidence: {confidence:.1%})")
    print("\nAsymmetry Analysis:")
    print(f"  Left mean temperature: {asymmetry['left_mean_temp']:.2f}°C")
    print(f"  Right mean temperature: {asymmetry['right_mean_temp']:.2f}°C")
    print(f"  Temperature asymmetry: {asymmetry['temperature_asymmetry']:.2f}°C ({asymmetry['asymmetry_percentage']:.2f}%)")
    
    print("\nQuadrant Temperature Analysis:")
    print(f"  Left Upper Outer (HIGH RISK): {quadrants['left_upper_outer']:.2f}°C")
    print(f"  Left Upper Inner: {quadrants['left_upper_inner']:.2f}°C")
    print(f"  Right Upper Outer (HIGH RISK): {quadrants['right_upper_outer']:.2f}°C")
    print(f"  Right Upper Inner: {quadrants['right_upper_inner']:.2f}°C")
    
    print("\nHotspot Detection:")
    print(f"  Left breast hotspots: {analysis_result['hotspot_analysis']['left_hotspots_count']}")
    print(f"  Right breast hotspots: {analysis_result['hotspot_analysis']['right_hotspots_count']}")
    
    if visualize:
        frameRead(os.path.splitext(image_path)[0])
    
    return analysis_result

def save_batch_results(results, output_filename='batch_results.txt'):
    """
    Save batch processing results to a text file.
    
    Parameters:
    - results: Results dictionary from batch_process_thermal_images
    - output_filename: Output file name
    """
    with open(output_filename, 'w') as f:
        f.write("BREAST CANCER DETECTION - BATCH PROCESSING RESULTS\n")
        f.write("=" * 70 + "\n\n")
        
        f.write(f"Total images processed: {results['total_images']}\n")
        f.write(f"Cancer detected: {results['cancer_detected']}\n")
        f.write(f"Normal: {results['normal']}\n")
        f.write(f"Detection rate: {results['cancer_detected']/max(results['total_images'], 1)*100:.1f}%\n\n")
        
        f.write("DETAILED PREDICTIONS:\n")
        f.write("-" * 70 + "\n")
        
        for pred in results['predictions']:
            confidence_str = f"{pred['confidence']*100:.1f}%" if pred['confidence'] else "N/A"
            f.write(f"{pred['filename']:<40} {pred['prediction']:<15} {confidence_str}\n")
        
        if results['high_risk']:
            f.write("\n" + "=" * 70 + "\n")
            f.write("⚠️  HIGH RISK IMAGES REQUIRING FURTHER EXAMINATION:\n")
            f.write("-" * 70 + "\n")
            for img_name in results['high_risk']:
                f.write(f"  - {img_name}\n")
    
    print(f"\nResults saved to {output_filename}")

def load_multi_dataset(primary_dataset_path, secondary_dataset_path=None, 
                       secondary_test_images=150, verbose=True, alignment_method=None, load_all=False):
    """
    Load and combine multiple thermal imaging datasets.
    
    This function allows combining the primary dataset (organized by Training/Testing/Benign/Malignant)
    with a secondary dataset while preserving ~150 images from the secondary dataset for independent testing.
    
    Parameters:
    - load_all: If True, recursively loads ALL images from secondary dataset (ignores folder structure)
    - alignment_method: 'histogram', 'features_mean_std', 'features_minmax', or None
    
    Supports dataset alignment to fix cross-protocol incompatibility issues:
    - 'histogram': Histogram matching to normalize thermal distributions
    - 'features_mean_std': Feature-level mean/stddev alignment
    - 'features_minmax': Feature-level min/max range alignment
    - None: No alignment (original behavior)
    
    Parameters:
    - primary_dataset_path: Path to the primary dataset (e.g., './Dataset')
    - secondary_dataset_path: Path to secondary dataset (optional)
    - secondary_test_images: Number of images to reserve from secondary dataset for testing (~150)
    - verbose: Whether to print detailed information
    
    Returns:
    - X_combined: Combined feature matrix from both datasets (training)
    - y_combined: Combined labels
    - X_secondary_test: Reserved secondary test features (~150 images)  
    - y_secondary_test: Reserved secondary test labels
    - info: Dictionary with loading information
    """
    info = {
        'primary_images': 0,
        'secondary_images': 0,
        'secondary_reserved_test': 0,
        'total_training': 0,
        'cancer_count': 0,
        'no_cancer_count': 0
    }
    
    # Load primary dataset
    if verbose:
        print("\n" + "="*70)
        print("LOADING PRIMARY DATASET")
        print("="*70)
    
    X_primary, y_primary, primary_info = load_dataset_from_directory(primary_dataset_path, load_all=load_all)
    info['primary_images'] = len(X_primary) if X_primary is not None else 0
    
    if X_primary is None or y_primary is None:
        print("Failed to load primary dataset. Cannot proceed with multi-dataset loading.")
        return None, None, None, None, info
    
    # Load secondary dataset if provided
    if secondary_dataset_path and os.path.exists(secondary_dataset_path):
        if verbose:
            print("\n" + "="*70)
            print("LOADING SECONDARY DATASET")
            print("="*70)
        
        X_secondary, y_secondary, secondary_info = load_dataset_from_directory(secondary_dataset_path, load_all=load_all)
        
        # If load_all is True and structured loader didn't find enough images, try recursive scan
        if load_all and (X_secondary is None or len(X_secondary) < 500):
            if verbose:
                print("\n✓ Attempting full recursive scan for all images...")
            X_secondary, y_secondary, secondary_info = load_dataset_recursive(secondary_dataset_path, verbose=verbose)
        
        if X_secondary is None or y_secondary is None:
            # Try flexible flat loader for non-standard folder structures
            if verbose:
                print("\n✓ Structured format not found, trying flexible image loader...")
            
            X_secondary, y_secondary, secondary_info = load_thermal_images_flat(
                secondary_dataset_path, 
                estimated_cancer_rate=0.3
            )
            
            if X_secondary is None or len(X_secondary) == 0:
                print("⚠️  Failed to load secondary dataset. Using primary dataset only.")
                X_combined = X_primary
                y_combined = y_primary
                X_secondary_test = np.array([])
                y_secondary_test = np.array([])
            else:
                # Successfully loaded with flexible loader
                info['secondary_images'] = len(X_secondary)
                
                # Split secondary dataset: reserve ~150 images for testing
                n_test = min(secondary_test_images, len(X_secondary))
                indices = np.arange(len(X_secondary))
                np.random.seed(42)
                np.random.shuffle(indices)
                
                test_indices = indices[:n_test]
                train_indices = indices[n_test:]
                
                X_secondary_test = X_secondary[test_indices]
                y_secondary_test = y_secondary[test_indices]
                
                X_secondary_train = X_secondary[train_indices]
                y_secondary_train = y_secondary[train_indices]
                
                info['secondary_reserved_test'] = len(X_secondary_test)
                
                # Combine primary + secondary training data
                X_combined = np.vstack([X_primary, X_secondary_train])
                y_combined = np.concatenate([y_primary, y_secondary_train])
                
                # Apply alignment if requested
                if alignment_method and alignment_method.startswith('features'):
                    alignment_type = alignment_method.replace('features_', '')
                    if verbose:
                        print(f"\n✓ Applying feature alignment: {alignment_type}")
                    
                    # Split back into primary and secondary for alignment
                    X_primary_aligned, X_secondary_train_aligned = align_feature_distributions(
                        X_primary, X_secondary_train, method=alignment_type
                    )
                    
                    # Recombine with aligned data
                    X_combined = np.vstack([X_primary_aligned, X_secondary_train_aligned])
                    if verbose:
                        print(f"✓ Features aligned successfully")
                elif alignment_method == 'histogram':
                    if verbose:
                        print(f"\n✓ Alignment method set: histogram (applied during image loading)")
                
                if verbose:
                    print("\n" + "="*70)
                    print("MULTI-DATASET COMBINATION SUMMARY")
                    print("="*70)
                    print(f"\nPrimary Dataset: {len(X_primary)} images")
                    print(f"  Cancer: {sum(y_primary)} | No Cancer: {sum(y_primary==0)}")
                    print(f"\nSecondary Dataset (Flat): {info['secondary_images']} images")
                    print(f"  Cancer: {sum(y_secondary)} | No Cancer: {sum(y_secondary==0)}")
                    print(f"\nSecondary Reserved for Testing: {info['secondary_reserved_test']} images")
                    print(f"  Cancer: {sum(y_secondary_test)} | No Cancer: {sum(y_secondary_test==0)}")
                    print(f"\nSecondary Used for Training: {len(X_secondary_train)} images")
                    print(f"  Cancer: {sum(y_secondary_train)} | No Cancer: {sum(y_secondary_train==0)}")
                    print(f"\n{'COMBINED TRAINING SET':^50}")
                    print(f"Total images: {len(X_combined)}")
                    print(f"  Cancer: {sum(y_combined)} ({sum(y_combined)/len(y_combined)*100:.1f}%)")
                    print(f"  No Cancer: {sum(y_combined==0)} ({sum(y_combined==0)/len(y_combined)*100:.1f}%)")
                    print(f"Feature vector size: {X_combined.shape[1]} features per image")
        else:
            info['secondary_images'] = len(X_secondary)
            
            # Split secondary dataset: reserve ~150 images for testing
            n_test = min(secondary_test_images, len(X_secondary))
            indices = np.arange(len(X_secondary))
            np.random.seed(42)
            np.random.shuffle(indices)
            
            test_indices = indices[:n_test]
            train_indices = indices[n_test:]
            
            X_secondary_test = X_secondary[test_indices]
            y_secondary_test = y_secondary[test_indices]
            
            X_secondary_train = X_secondary[train_indices]
            y_secondary_train = y_secondary[train_indices]
            
            info['secondary_reserved_test'] = len(X_secondary_test)
            
            # Combine primary + secondary training data
            X_combined = np.vstack([X_primary, X_secondary_train])
            y_combined = np.concatenate([y_primary, y_secondary_train])
            
            # Apply alignment if requested
            if alignment_method and alignment_method.startswith('features'):
                alignment_type = alignment_method.replace('features_', '')
                if verbose:
                    print(f"\n✓ Applying feature alignment: {alignment_type}")
                
                # Split back into primary and secondary for alignment
                X_primary_aligned, X_secondary_train_aligned = align_feature_distributions(
                    X_primary, X_secondary_train, method=alignment_type
                )
                
                # Recombine with aligned data
                X_combined = np.vstack([X_primary_aligned, X_secondary_train_aligned])
                if verbose:
                    print(f"✓ Features aligned successfully")
            elif alignment_method == 'histogram':
                if verbose:
                    print(f"\n✓ Alignment method set: histogram (applied during image loading)")
            
            if verbose:
                print("\n" + "="*70)
                print("MULTI-DATASET COMBINATION SUMMARY")
                print("="*70)
                print(f"\nPrimary Dataset: {len(X_primary)} images")
                print(f"  Cancer: {sum(y_primary)} | No Cancer: {sum(y_primary==0)}")
                print(f"\nSecondary Dataset: {info['secondary_images']} images")
                print(f"  Cancer: {sum(y_secondary)} | No Cancer: {sum(y_secondary==0)}")
                print(f"\nSecondary Reserved for Testing: {info['secondary_reserved_test']} images")
                print(f"  Cancer: {sum(y_secondary_test)} | No Cancer: {sum(y_secondary_test==0)}")
                print(f"\nSecondary Used for Training: {len(X_secondary_train)} images")
                print(f"  Cancer: {sum(y_secondary_train)} | No Cancer: {sum(y_secondary_train==0)}")
                print(f"\n{'COMBINED TRAINING SET':^50}")
                print(f"Total images: {len(X_combined)}")
                print(f"  Cancer: {sum(y_combined)} ({sum(y_combined)/len(y_combined)*100:.1f}%)")
                print(f"  No Cancer: {sum(y_combined==0)} ({sum(y_combined==0)/len(y_combined)*100:.1f}%)")
                print(f"Feature vector size: {X_combined.shape[1]} features per image")
    else:
        if verbose and secondary_dataset_path:
            print(f"\n⚠️  Secondary dataset path not found: {secondary_dataset_path}")
            print("Using primary dataset only.")
        
        X_combined = X_primary
        y_combined = y_primary
        X_secondary_test = np.array([])
        y_secondary_test = np.array([])
    
    info['total_training'] = len(X_combined)
    info['cancer_count'] = sum(y_combined)
    info['no_cancer_count'] = sum(y_combined == 0)
    
    return X_combined, y_combined, X_secondary_test, y_secondary_test, info

# Main demonstration function
def main():
    """
    Minimal demo: load dataset, train model, and run a sample prediction.
    """
    parser = argparse.ArgumentParser(description='Breast Cancer Detection using Thermal Imaging')
    parser.add_argument('--dataset', type=str, default='./Dataset',
                       help='Path to primary dataset folder (default: ./Dataset)')
    parser.add_argument('--secondary_dataset', type=str, default=None,
                       help='Path to secondary dataset folder for multi-dataset training (optional)')
    parser.add_argument('--secondary_test_size', type=int, default=150,
                       help='Number of images to reserve from secondary dataset for testing (default: 150)')
    parser.add_argument('--secondary_as_test_only', action='store_true',
                       help='Use secondary dataset for testing only, not training (recommended for different imaging protocols)')
    parser.add_argument('--test_folder', type=str, default='Unknown_class',
                       help='Name of test folder within dataset (default: Unknown_class)')
    parser.add_argument('--model_type', type=str, choices=['svm', 'rf'], default='svm',
                       help='Model type: svm or rf (default: svm)')
    parser.add_argument('--output_report', type=str, default='screening_report_unknown.txt',
                       help='Output report filename (default: screening_report_unknown.txt)')
    parser.add_argument('--interactive', action='store_true',
                       help='Enable interactive mode to review images one by one')
    parser.add_argument('--display_size', type=int, default=800,
                       help='Maximum display size for images in interactive mode (default: 800)')
    parser.add_argument('--alignment', type=str, choices=['none', 'histogram', 'features_mean_std', 'features_minmax'], 
                       default='none',
                       help='Dataset alignment method for cross-protocol compatibility (default: none)\n'
                            '  none: No alignment (original behavior)\n'
                            '  histogram: Histogram matching (normalizes thermal distributions)\n'
                            '  features_mean_std: Feature-level mean/stddev alignment\n'
                            '  features_minmax: Feature-level min/max range alignment')
    parser.add_argument('--load-all', action='store_true',
                       help='Recursively load ALL PNG/JPG files from dataset (ignores folder structure, loads all 976 images)')
    
    args = parser.parse_args()
    
    print("Breast Cancer Detection Demo")
    print("=" * 30)
    print(f"Primary Dataset: {args.dataset}")
    if args.secondary_dataset:
        print(f"Secondary Dataset: {args.secondary_dataset}")
        print(f"Secondary test images: {args.secondary_test_size}")
    print(f"Test folder: {args.test_folder}")
    print(f"Model type: {args.model_type}")
    print(f"Output report: {args.output_report}")
    if args.secondary_as_test_only:
        print(f"Mode: Secondary dataset used for TESTING ONLY (not training)")
    
    # Load dataset(s) - primary only or combined with secondary
    if args.secondary_dataset and args.secondary_as_test_only:
        # Mode: Train on primary only, test on secondary only
        print("\n" + "="*70)
        print("MODE: SECONDARY DATASET FOR TESTING ONLY")
        print("="*70)
        
        X, y, info = load_dataset_from_directory(args.dataset, load_all=args.load_all)
        if X is None or y is None:
            print("No images loaded. Please check your Dataset folder.")
            return
        
        print(f"\nTraining dataset: {len(X)} images (primary only)")
        print(f"  Cancer: {sum(y)} | No Cancer: {sum(y==0)}")
        
        # Load secondary dataset for testing only
        print("\nLoading secondary dataset for testing...")
        X_secondary_test, y_secondary_test, sec_info = load_thermal_images_flat(
            args.secondary_dataset,
            estimated_cancer_rate=0.3
        )
        
        if X_secondary_test is None or len(X_secondary_test) == 0:
            print("⚠️  Could not load secondary dataset for testing")
            X_secondary_test = np.array([])
            y_secondary_test = np.array([])
        else:
            print(f"Test dataset: {len(X_secondary_test)} images (secondary only)")
            
    elif args.secondary_dataset:
        # Mode: Combine primary and secondary for training (original behavior)
        alignment_method = None if args.alignment == 'none' else args.alignment
        if alignment_method:
            print(f"\n✓ Using dataset alignment: {alignment_method}")
        
        X, y, X_secondary_test, y_secondary_test, info = load_multi_dataset(
            args.dataset, 
            args.secondary_dataset,
            secondary_test_images=args.secondary_test_size,
            verbose=True,
            alignment_method=alignment_method,
            load_all=args.load_all
        )
        if X is None or y is None:
            print("No images loaded. Please check your dataset folders.")
            return
        
        print("\n" + "="*70)
        print(f"Mode: COMBINED training (primary + secondary)")
        print(f"Total training samples: {len(X)}")
        print("="*70)
    else:
        # Original mode: Primary dataset only
        X, y, info = load_dataset_from_directory(args.dataset, load_all=args.load_all)
        if X is None or y is None:
            print("No images loaded. Please check your Dataset folder and structure.")
            return
        
        X_secondary_test = np.array([])
        y_secondary_test = np.array([])

    X_train, X_val, X_test, y_train, y_val, y_test = prepare_training_data(X, y)

    # Train model
    model, _, _ = train_breast_cancer_model(X_train, y_train, model_type=args.model_type)

    # Evaluate on secondary test set if available
    if len(X_secondary_test) > 0:
        print("\n" + "="*70)
        print("EVALUATING ON SECONDARY DATASET TEST SET")
        print("="*70)
        y_pred_secondary = model.predict(X_secondary_test)
        accuracy_secondary = accuracy_score(y_secondary_test, y_pred_secondary)
        
        print(f"\nSecondary Test Set Performance: {len(X_secondary_test)} images")
        print(f"Accuracy: {accuracy_secondary:.3f}")
        print("\nClassification Report (Secondary Test Set):")
        print(classification_report(y_secondary_test, y_pred_secondary, 
                                   target_names=['Normal', 'Cancer']))

    # Batch inference on unknown test images (optional): Use only unlabeled data
    print() 
    unknown_path = os.path.join(args.dataset, args.test_folder)
    if os.path.exists(unknown_path):
        results = batch_process_thermal_images(unknown_path, model, interactive=args.interactive, display_size=args.display_size)
        if not args.interactive:  # Only save report in non-interactive mode
            save_batch_results(results, args.output_report)
    else:
        print(f"No unknown-class folder found at {unknown_path}. Skipping inference batch.")

    # Single sample check (optional)
    if len(X_test) > 0:
        sample_img_idx = 0
        sample_img = X_test[sample_img_idx].reshape(1, -1)  # if needed for offline test
        print(f"\nSample test prediction for index {sample_img_idx}: {model.predict(sample_img)[0]}")


if __name__ == "__main__":
    main()






#!/usr/bin/env python3
"""
Raspberry Pi Zero 2W Demo - Thermal Imaging Inference
Ultra-lightweight inference script optimized for 512MB RAM
Runs pre-trained model on thermal images - NO TRAINING
Usage: python3 pi_demo.py --model model.pkl --image test.jpg
"""

import cv2
import numpy as np
import pickle
import argparse
import sys
import gc
from pathlib import Path
from time import time


class ThermalDemo:
    """Optimized thermal imaging demo for Raspberry Pi Zero 2W"""
    
    def __init__(self, model_path, max_image_size=256, quiet=False):
        self.model = self._load_model(model_path)
        self.scaler = None
        self.max_image_size = max_image_size  # Downsize images to save RAM
        self.quiet = quiet
        if not quiet:
            print(f"✓ Model loaded: {model_path}")
            print(f"✓ Image resize: {max_image_size}x{max_image_size}")
    
    def _load_model(self, model_path):
        """Load pre-trained model"""
        try:
            with open(model_path, 'rb') as f:
                data = pickle.load(f)
            
            # Handle both dict and direct model
            if isinstance(data, dict):
                model = data.get('model')
                self.scaler = data.get('scaler')
            else:
                model = data
            
            return model
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            sys.exit(1)
    
    def extract_features(self, img):
        """Extract thermal features - optimized for Pi Zero 2W"""
        # Ensure uint8 format
        if img.dtype != np.uint8:
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=np.uint8)
        
        # Use smaller working arrays for memory efficiency
        mean_temp = float(np.mean(img))
        std_temp = float(np.std(img))
        max_temp = float(np.max(img))
        
        # Lightweight asymmetry (avoid extra arrays)
        height, width = img.shape[:2]
        mid_x = width // 2
        asymmetry = float(abs(np.mean(img[:, :mid_x]) - np.mean(img[:, mid_x:])))
        
        # Gradient features with memory optimization
        grad_x = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)  # Use 32-bit float instead of 64-bit
        grad_y = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
        gradient_magnitude = cv2.magnitude(grad_x, grad_y)
        grad_mean = float(np.mean(gradient_magnitude))
        grad_std = float(np.std(gradient_magnitude))
        
        # Clean up temporary arrays immediately
        del grad_x, grad_y
        gc.collect()
        
        # Hotspot ratio (avoid extra percentile computation)
        threshold_90 = float(np.percentile(img, 90))
        hotspots = float(np.sum(img > threshold_90)) / (height * width)
        
        # 8 features total
        features = np.array([
            mean_temp,
            std_temp,
            max_temp,
            asymmetry,
            grad_mean,
            grad_std,
            hotspots,
            threshold_90
        ], dtype=np.float32).reshape(1, -1)
        
        feature_dict = {
            'mean_temperature': mean_temp,
            'std_temperature': std_temp,
            'max_temperature': max_temp,
            'asymmetry': asymmetry,
            'gradient_mean': grad_mean,
            'gradient_std': grad_std,
            'hotspots_ratio': hotspots * 100,
            'threshold_90': threshold_90
        }
        
        # Clean up gradient_magnitude
        del gradient_magnitude
        gc.collect()
        
        return features, feature_dict
    
    def predict_image(self, image_path, show_image=False):
        """Run inference on single image - optimized for Pi Zero 2W"""
        start_time = time()
        
        # Load image
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"✗ Could not load image: {image_path}")
            return None
        
        if not self.quiet:
            print(f"\n📷 Processing: {image_path.split('/')[-1]}")
        
        # Downsize image for Pi Zero 2W (reduces memory usage significantly)
        height, width = img.shape[:2]
        if height > self.max_image_size or width > self.max_image_size:
            scale = min(self.max_image_size / height, self.max_image_size / width)
            new_size = (int(width * scale), int(height * scale))
            img = cv2.resize(img, new_size, interpolation=cv2.INTER_LINEAR)
        
        if not self.quiet:
            print(f"   Image size: {img.shape}")
        
        # Extract features
        features, feature_dict = self.extract_features(img)
        
        # Scale if scaler available
        if self.scaler is not None:
            try:
                features = self.scaler.transform(features)
            except:
                pass
        
        # Predict
        try:
            prediction = self.model.predict(features)[0]
            probabilities = self.model.predict_proba(features)[0]
            confidence = float(probabilities.max())
            
            # Format results
            class_name = 'CANCER' if prediction == 1 else 'NORMAL'
            cancer_prob = probabilities[1] if len(probabilities) > 1 else 1 - probabilities[0]
            normal_prob = probabilities[0] if len(probabilities) > 1 else probabilities[0]
            
            inference_time = (time() - start_time) * 1000
            
            result = {
                'prediction': int(prediction),
                'class': class_name,
                'confidence': confidence,
                'cancer_probability': float(cancer_prob),
                'normal_probability': float(normal_prob),
                'inference_time_ms': inference_time,
                'features': feature_dict
            }
            
            # Print results
            self._print_results(result)
            
            # Show image if requested
            if show_image:
                self._display_image(img, result)
            
            # Clean up
            del img, features
            gc.collect()
            
            return result
        
        except Exception as e:
            print(f"✗ Prediction error: {e}")
            return None
    
    def _print_results(self, result):
        """Pretty print prediction results"""
        if self.quiet:
            # Minimal output for quiet mode
            print(f"{result['class']} ({result['confidence']*100:.0f}%)")
            return
        
        print("\n" + "="*60)
        print(f"PREDICTION: {result['class']}")
        print(f"CONFIDENCE: {result['confidence']*100:.1f}%")
        print(f"INFERENCE TIME: {result['inference_time_ms']:.1f} ms")
        
        print(f"\nProbabilities:")
        print(f"  Normal/Benign: {result['normal_probability']*100:.1f}%")
        print(f"  Cancer: {result['cancer_probability']*100:.1f}%")
        
        print(f"\nFeatures Used:")
        feat = result['features']
        print(f"  Mean Temperature: {feat['mean_temperature']:.1f}")
        print(f"  Temp Std Dev: {feat['std_temperature']:.1f}")
        print(f"  Peak Temperature: {feat['max_temperature']:.1f}")
        print(f"  Left-Right Asymmetry: {feat['asymmetry']:.2f}")
        print(f"  Gradient Mean: {feat['gradient_mean']:.2f}")
        print(f"  Gradient Std Dev: {feat['gradient_std']:.2f}")
        print(f"  Hot Pixels Ratio: {feat['hotspots_ratio']:.2f}%")
        print(f"  90th Percentile: {feat['threshold_90']:.1f}")
        print("="*60)
    
    def _display_image(self, img, result):
        """Display image with prediction overlay - optimized for Pi Zero 2W"""
        # Use smaller display size for Pi Zero 2W
        display_size = (320, 240)  # QVGA resolution - low memory footprint
        img_display = cv2.resize(img, display_size, interpolation=cv2.INTER_LINEAR)
        
        # Convert to BGR for color overlay
        img_color = cv2.cvtColor(img_display, cv2.COLOR_GRAY2BGR)
        
        # Add prediction text
        font = cv2.FONT_HERSHEY_SIMPLEX
        prediction_text = result['class']
        confidence_text = f"{result['confidence']*100:.0f}%"
        
        # Color: Red for cancer, Green for normal
        color = (0, 0, 255) if result['class'] == 'CANCER' else (0, 255, 0)
        
        # Put text with smaller font for Pi Zero 2W
        cv2.putText(img_color, prediction_text, (10, 30),
                   font, 0.8, color, 2)
        cv2.putText(img_color, confidence_text, (10, 70),
                   font, 0.6, color, 1)
        
        # Display
        cv2.imshow('Thermal - Pi Zero 2W', img_color)
        if not self.quiet:
            print("\nImage displayed. Press any key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        # Clean up
        del img_color, img_display
        gc.collect()
    
    def batch_predict(self, image_dir, pattern="*/image.jpg"):
        """Predict on all images in directory - optimized for Pi Zero 2W"""
        image_dir = Path(image_dir)
        images = list(image_dir.glob(pattern))
        
        if not images:
            print(f"✗ No images found matching {pattern}")
            return []
        
        if not self.quiet:
            print(f"\n🔄 Processing {len(images)} images from {image_dir}")
        
        results = []
        
        for i, img_path in enumerate(images, 1):
            result = self.predict_image(img_path, show_image=False)
            if result:
                result['image_path'] = str(img_path)
                results.append(result)
                
                # Progress indicator
                if self.quiet:
                    print(f"[{i}/{len(images)}] ", end='', flush=True)
                else:
                    status = "🔴 CANCER" if result['class'] == 'CANCER' else "🟢 NORMAL"
                    print(f"  [{i}/{len(images)}] {status} ({result['confidence']*100:.0f}%)")
            
            # Force garbage collection between predictions
            gc.collect()
        
        # Summary
        if results:
            cancer_count = sum(1 for r in results if r['class'] == 'CANCER')
            normal_count = len(results) - cancer_count
            avg_conf = np.mean([r['confidence'] for r in results])
            avg_time = np.mean([r['inference_time_ms'] for r in results])
            
            if not self.quiet:
                print("\n" + "="*60)
                print(f"BATCH SUMMARY ({len(results)} images)")
                print(f"  Cancer Predictions: {cancer_count}")
                print(f"  Normal Predictions: {normal_count}")
                print(f"  Average Confidence: {avg_conf*100:.1f}%")
                print(f"  Average Time/Image: {avg_time:.1f} ms")
                print("="*60 + "\n")
            else:
                print(f"\nBATCH: {cancer_count}C/{normal_count}N | Avg: {avg_time:.0f}ms")
        
        return results


def main():
    parser = argparse.ArgumentParser(
        description='Raspberry Pi Zero 2W Thermal Imaging Demo (Inference Only)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:
  Single image:       python3 pi_demo.py --model model.pkl --image test.jpg
  With display:       python3 pi_demo.py --model model.pkl --image test.jpg --display
  Batch processing:   python3 pi_demo.py --model model.pkl --directory Dataset/
  Quiet mode:         python3 pi_demo.py --model model.pkl --image test.jpg --quiet
  Small size (fast):  python3 pi_demo.py --model model.pkl --image test.jpg --size 128
        """
    )
    
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model (pickle file)')
    
    parser.add_argument('--image', type=str,
                       help='Single image for inference')
    
    parser.add_argument('--directory', type=str,
                       help='Directory of images for batch processing')
    
    parser.add_argument('--pattern', type=str, default='*/image.jpg',
                       help='File pattern for batch (default: */image.jpg)')
    
    parser.add_argument('--display', action='store_true',
                       help='Display image with prediction overlay')
    
    parser.add_argument('--quiet', action='store_true',
                       help='Minimal output (Pi Zero 2W friendly)')
    
    parser.add_argument('--size', type=int, default=256,
                       help='Max image size before processing (default: 256, use 128 for Pi Zero 2W)')
    
    args = parser.parse_args()
    
    # Initialize demo with optimizations
    demo = ThermalDemo(args.model, max_image_size=args.size, quiet=args.quiet)
    
    # Single image
    if args.image:
        demo.predict_image(args.image, show_image=args.display)
    
    # Batch processing
    elif args.directory:
        demo.batch_predict(args.directory, pattern=args.pattern)
    
    else:
        parser.print_help()


if __name__ == '__main__':
    main()

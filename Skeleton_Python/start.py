#!/usr/bin/env python3
"""
Quick start script for CVPR Visual Search Coursework
Provides a simple menu-driven interface to run all components
"""

import os
import sys
import subprocess


def print_header():
    """Print the header."""
    print("\n" + "="*70)
    print("VISUAL SEARCH SYSTEM - CVPR COURSEWORK 2025")
    print("University of Surrey")
    print("="*70)


def print_menu():
    """Print the main menu."""
    print("\nMain Menu:")
    print("1. Install Dependencies")
    print("2. Compute Descriptors (Step 1 - Required)")
    print("3. Run Fusion Accuracy Evaluation (Step 2 - Best Results)")
    print("4. Run Single Query Visual Search")
    print("5. Run Quantization Comparison")
    print("6. Launch Interactive GUI (Optional)")
    print("8. Run Complete Pipeline (Steps 1-6)")
    print("0. Exit")
    print()


def install_dependencies():
    """Install required Python packages."""
    print("\n" + "-"*70)
    print("Installing dependencies...")
    print("-"*70)
    
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)
        print("\n✓ Dependencies installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Error installing dependencies: {e}")
        return False
    
    return True


def compute_descriptors():
    """Compute descriptors for all images."""
    print("\n" + "-"*70)
    print("Computing descriptors...")
    print("This will take approximately 5-10 minutes")
    print("-"*70)
    
    try:
        subprocess.run([sys.executable, "Skeleton_Python/core/cvpr_computedescriptors.py"], check=True)
        print("\n✓ Descriptors computed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Error computing descriptors: {e}")
        return False


def run_experiments():
    """Run all experiments."""
    print("\n" + "-"*70)
    print("Running fusion accuracy evaluation (Set 15)...")
    print("-"*70)
    
    if not os.path.exists('./descriptors'):
        print("\n✗ Error: Descriptors not found!")
        print("Please run 'Compute Descriptors' first (Option 2)")
        return False
    
    try:
        subprocess.run([sys.executable, "Skeleton_Python/advanced/test_fusion_query_expansion.py"], check=True)
        print("\n✓ Fusion accuracy evaluation completed!")
        print("Results saved to ./results/ folder")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Error running fusion accuracy evaluation: {e}")
        return False


def launch_gui():
    """Launch the interactive GUI."""
    print("\n" + "-"*70)
    print("Launching interactive GUI...")
    print("-"*70)
    
    if not os.path.exists('./descriptors'):
        print("\n✗ Error: Descriptors not found!")
        print("Please run 'Compute Descriptors' first (Option 2)")
        return False
    
    try:
        subprocess.run([sys.executable, "Skeleton_Python/cvpr_visualsearch.py"])
        return True
    except Exception as e:
        print(f"\n✗ Error launching GUI: {e}")
        return False


def run_single_query():
    """Run single query visual search."""
    print("\n" + "-"*70)
    print("Running single query visual search...")
    print("-"*70)
    try:
        subprocess.run([sys.executable, "Skeleton_Python/single_query_visual_search.py"], check=True)
        print("\n✓ Single query visual search completed!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Error running single query visual search: {e}")
        return False


def run_quantization_comparison():
    """Run quantization comparison."""
    print("\n" + "-"*70)
    print("Running quantization comparison...")
    print("-"*70)
    try:
        subprocess.run([sys.executable, "Skeleton_Python/experiments/plot_quantization_comparison.py"], check=True)
        print("\n✓ Quantization comparison completed!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Error running quantization comparison: {e}")
        return False


def run_complete_pipeline():
    """Run the complete pipeline."""
    print("\n" + "="*70)
    print("RUNNING COMPLETE PIPELINE")
    print("="*70)
    
    # Step 1: Compute descriptors
    print("\n[1/6] Computing Descriptors...")
    if not compute_descriptors():
        return False
    input("\nPress Enter to continue to fusion accuracy evaluation...")
    # Step 2: Run fusion accuracy evaluation
    print("\n[2/6] Running Fusion Accuracy Evaluation...")
    if not run_experiments():
        return False
    input("\nPress Enter to continue to single query visual search...")
    # Step 3: Run single query visual search
    print("\n[3/6] Running Single Query Visual Search...")
    if not run_single_query():
        return False
    input("\nPress Enter to continue to quantization comparison...")
    # Step 4: Run quantization comparison
    print("\n[4/6] Running Quantization Comparison...")
    if not run_quantization_comparison():
        return False
    input("\nPress Enter to continue to GUI...")
    # Step 5: Launch GUI
    print("\n[5/6] Launching Interactive GUI...")
    if not launch_gui():
        return False
    print("\n" + "="*70)
    print("PIPELINE COMPLETED!")
    print("="*70)
    print("\nYou can now:")
    print("  - Review results in ./results/ folder")
    print("  - Run the GUI (Option 6) for interactive testing")
    print("  - Analyze the PR curves and confusion matrices")
    return True


def check_dataset():
    """Check if dataset is available."""
    dataset_path = os.path.join(os.path.dirname(__file__), '../MSRC_ObjCategImageDatabase_v2/Images')
    
    if not os.path.exists(dataset_path):
        print("\n" + "!"*70)
        print("WARNING: MSRC Dataset not found!")
        print("!"*70)
        print(f"\nExpected location: {dataset_path}")
        print("\nPlease ensure the dataset is in the correct location:")
        print("  MSRC_ObjCategImageDatabase_v2/")
        print("    ├── Images/")
        print("    └── GroundTruth/")
        print()
        return False
    
    # Count images
    num_images = len([f for f in os.listdir(dataset_path) if f.endswith('.bmp')])
    print(f"\n✓ Dataset found: {num_images} images")
    
    return True


def main():
    """Main function."""
    print_header()
    
    # Check dataset
    if not check_dataset():
        print("\nPlease fix the dataset location and try again.")
        return
    
    while True:
        print_menu()
        
        try:
            choice = input("Enter your choice (0-6): ").strip()
            
            if choice == '0':
                print("\nThank you for using the Visual Search System!")
                print("Good luck with your coursework! 🎓")
                break
            
            elif choice == '1':
                install_dependencies()
            
            elif choice == '2':
                compute_descriptors()
            
            elif choice == '3':
                run_experiments()
            elif choice == '4':
                run_single_query()
            elif choice == '5':
                run_quantization_comparison()
            elif choice == '6':
                launch_gui()
            elif choice == '8':
                run_complete_pipeline()
            
            else:
                print("\n✗ Invalid choice. Please enter a number between 0 and 6.")
            
            input("\nPress Enter to continue...")
            
        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            print(f"\n✗ Unexpected error: {e}")
            input("\nPress Enter to continue...")


if __name__ == '__main__':
    main()

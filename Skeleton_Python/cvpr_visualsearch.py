"""
Visual search system with interactive GUI.
Supports multiple descriptor types, distance metrics, and PCA.
"""

import os
import sys
import numpy as np
import scipy.io as sio
import cv2
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
from cvpr_compare import batch_compare
from core.pca_utils import PCAProjection
from core.evaluation import get_class_from_filename

DESCRIPTOR_FOLDER = os.path.abspath('./descriptors')
DATASET_FOLDER = os.path.abspath('../MSRC_ObjCategImageDatabase_v2')
IMAGES_FOLDER = os.path.join(DATASET_FOLDER, 'Images')


class VisualSearchSystem:
    """
    Main visual search system class.
    """
    
    def __init__(self):
        self.descriptors = None
        self.filenames = None
        self.pca_model = None
        self.pca_descriptors = None
        self.current_descriptor_type = None
        self.current_distance_metric = 'euclidean'
        self.use_pca = False
        
    def load_descriptors(self, descriptor_type='global_hist_8'):
        """
        Load descriptors from disk.
        
        Args:
            descriptor_type: Name of descriptor subfolder
        """
        descriptor_path = os.path.join(DESCRIPTOR_FOLDER, descriptor_type)
        
        if not os.path.exists(descriptor_path):
            raise ValueError(f"Descriptor folder not found: {descriptor_path}")
        
        print(f"Loading descriptors from: {descriptor_path}")
        
        # Load all descriptors
        all_files = sorted([f for f in os.listdir(descriptor_path) if f.endswith('.mat')])
        
        if len(all_files) == 0:
            raise ValueError(f"No descriptor files found in {descriptor_path}")
        
        descriptors_list = []
        filenames_list = []
        
        for mat_file in all_files:
            mat_path = os.path.join(descriptor_path, mat_file)
            data = sio.loadmat(mat_path)
            descriptors_list.append(data['F'].flatten())
            
            # Get corresponding image filename
            img_filename = mat_file.replace('.mat', '.bmp')
            filenames_list.append(img_filename)
        
        self.descriptors = np.array(descriptors_list)
        self.filenames = np.array(filenames_list)
        self.current_descriptor_type = descriptor_type
        self.pca_model = None
        self.pca_descriptors = None
        self.use_pca = False
        
        print(f"Loaded {len(self.descriptors)} descriptors")
        print(f"Descriptor dimensionality: {self.descriptors.shape[1]}")
        
    def apply_pca(self, n_components=None, variance_threshold=0.95):
        """
        Apply PCA to loaded descriptors.
        
        Args:
            n_components: Number of components to keep
            variance_threshold: Variance to preserve
        """
        if self.descriptors is None:
            raise ValueError("Load descriptors first")
        
        print(f"\nApplying PCA...")
        self.pca_model = PCAProjection(n_components, variance_threshold)
        self.pca_descriptors = self.pca_model.fit_transform(self.descriptors)
        self.pca_model.compute_covariance_inverse()
        
        print(f"Reduced dimensionality: {self.descriptors.shape[1]} -> {self.pca_descriptors.shape[1]}")
        
    def search(self, query_idx, distance_metric='euclidean', use_pca=False, top_k=20):
        """
        Perform visual search.
        
        Args:
            query_idx: Index of query image
            distance_metric: Distance metric to use
            use_pca: Whether to use PCA-transformed descriptors
            top_k: Number of top results to return
        
        Returns:
            Tuple of (indices, distances, filenames)
        """
        if use_pca and self.pca_descriptors is None:
            raise ValueError("PCA not applied. Call apply_pca() first.")
        
        # Select descriptors
        if use_pca:
            descriptors = self.pca_descriptors
            query = self.pca_descriptors[query_idx]
        else:
            descriptors = self.descriptors
            query = self.descriptors[query_idx]
        
        # Compute distances
        if distance_metric == 'mahalanobis' and use_pca:
            cov_inv = self.pca_model.get_mahalanobis_cov_inv()
            distances = batch_compare(query, descriptors, distance_metric, cov_inv)
        else:
            distances = batch_compare(query, descriptors, distance_metric)
        
        # Sort by distance
        sorted_indices = np.argsort(distances)[:top_k]
        sorted_distances = distances[sorted_indices]
        sorted_filenames = self.filenames[sorted_indices]
        
        return sorted_indices, sorted_distances, sorted_filenames


class VisualSearchGUI:
    """
    GUI for visual search system.
    """
    
    def __init__(self, root):
        self.root = root
        self.root.title("Visual Search System - CVPR Coursework")
        self.root.geometry("1400x900")
        
        self.search_system = VisualSearchSystem()
        self.query_idx = None
        self.result_images = []
        
        self.setup_ui()
        
    def setup_ui(self):
        """
        Setup the user interface.
        """
        # Control panel
        control_frame = ttk.Frame(self.root, padding="10")
        control_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Descriptor selection
        ttk.Label(control_frame, text="Descriptor Type:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.descriptor_var = tk.StringVar(value='global_hist_8')
        descriptor_combo = ttk.Combobox(control_frame, textvariable=self.descriptor_var, width=25)
        descriptor_combo['values'] = [
            'global_hist_4', 'global_hist_8', 'global_hist_16',
            'grid4x4_color',
            'texture_12orient'
        ]
        descriptor_combo.grid(row=0, column=1, sticky=tk.W, pady=5, padx=5)
        
        # Distance metric selection
        ttk.Label(control_frame, text="Distance Metric:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.distance_var = tk.StringVar(value='euclidean')
        distance_combo = ttk.Combobox(control_frame, textvariable=self.distance_var, width=25)
        distance_combo['values'] = ['euclidean', 'l1', 'cosine', 'chi_square', 'intersection', 'mahalanobis']
        distance_combo.grid(row=1, column=1, sticky=tk.W, pady=5, padx=5)
        
        # PCA options
        self.use_pca_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(control_frame, text="Use PCA", variable=self.use_pca_var).grid(row=2, column=0, sticky=tk.W, pady=5)
        
        ttk.Label(control_frame, text="PCA Components:").grid(row=2, column=1, sticky=tk.W, pady=5)
        self.pca_components_var = tk.StringVar(value='50')
        ttk.Entry(control_frame, textvariable=self.pca_components_var, width=10).grid(row=2, column=2, sticky=tk.W, pady=5, padx=5)
        
        # Buttons
        ttk.Button(control_frame, text="Load Descriptors", command=self.load_descriptors).grid(row=3, column=0, pady=10, padx=5)
        ttk.Button(control_frame, text="Select Random Query", command=self.select_random_query).grid(row=3, column=1, pady=10, padx=5)
        ttk.Button(control_frame, text="Search", command=self.perform_search).grid(row=3, column=2, pady=10, padx=5)
        
        # Query image display
        query_frame = ttk.LabelFrame(self.root, text="Query Image", padding="10")
        query_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=10, pady=10)
        
        self.query_label = ttk.Label(query_frame)
        self.query_label.pack()
        
        self.query_info_label = ttk.Label(query_frame, text="No query selected")
        self.query_info_label.pack()
        
        # Results display
        results_frame = ttk.LabelFrame(self.root, text="Search Results (Top 20)", padding="10")
        results_frame.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=10, pady=10)
        
        # Create canvas with scrollbar for results
        canvas = tk.Canvas(results_frame, height=500)
        scrollbar = ttk.Scrollbar(results_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        self.results_frame = scrollable_frame
        
        # Status bar
        self.status_label = ttk.Label(self.root, text="Ready", relief=tk.SUNKEN)
        self.status_label.grid(row=3, column=0, sticky=(tk.W, tk.E), padx=10, pady=5)
        
    def load_descriptors(self):
        """
        Load descriptors based on selected type.
        """
        descriptor_type = self.descriptor_var.get()
        
        try:
            self.status_label.config(text=f"Loading descriptors: {descriptor_type}...")
            self.root.update()
            
            self.search_system.load_descriptors(descriptor_type)
            
            # Apply PCA if requested
            if self.use_pca_var.get():
                try:
                    n_components = int(self.pca_components_var.get())
                except:
                    n_components = None
                
                self.search_system.apply_pca(n_components=n_components)
            
            self.status_label.config(text=f"Loaded {len(self.search_system.descriptors)} descriptors")
            messagebox.showinfo("Success", f"Loaded {len(self.search_system.descriptors)} descriptors")
            
        except Exception as e:
            self.status_label.config(text="Error loading descriptors")
            messagebox.showerror("Error", str(e))
    
    def select_random_query(self):
        """
        Select a random query image.
        """
        if self.search_system.descriptors is None:
            messagebox.showwarning("Warning", "Load descriptors first")
            return
        
        self.query_idx = np.random.randint(0, len(self.search_system.descriptors))
        query_filename = self.search_system.filenames[self.query_idx]
        
        # Display query image
        img_path = os.path.join(IMAGES_FOLDER, query_filename)
        img = Image.open(img_path)
        img.thumbnail((200, 200))
        photo = ImageTk.PhotoImage(img)
        
        self.query_label.config(image=photo)
        self.query_label.image = photo
        
        query_class = get_class_from_filename(query_filename)
        self.query_info_label.config(text=f"{query_filename}\nClass: {query_class}")
        
        self.status_label.config(text=f"Query selected: {query_filename}")
    
    def perform_search(self):
        """
        Perform visual search with current settings.
        """
        if self.query_idx is None:
            messagebox.showwarning("Warning", "Select a query image first")
            return
        
        try:
            distance_metric = self.distance_var.get()
            use_pca = self.use_pca_var.get()
            
            # Check if Mahalanobis with PCA
            if distance_metric == 'mahalanobis' and not use_pca:
                messagebox.showwarning("Warning", "Mahalanobis distance requires PCA. Enabling PCA...")
                use_pca = True
                if self.search_system.pca_descriptors is None:
                    try:
                        n_components = int(self.pca_components_var.get())
                    except:
                        n_components = None
                    self.search_system.apply_pca(n_components=n_components)
            
            self.status_label.config(text="Searching...")
            self.root.update()
            
            # Perform search
            indices, distances, filenames = self.search_system.search(
                self.query_idx, distance_metric, use_pca, top_k=20
            )
            
            # Display results
            self.display_results(indices, distances, filenames)
            
            self.status_label.config(text=f"Search completed. Distance metric: {distance_metric}")
            
        except Exception as e:
            self.status_label.config(text="Error during search")
            messagebox.showerror("Error", str(e))
            import traceback
            traceback.print_exc()
    
    def display_results(self, indices, distances, filenames):
        """
        Display search results.
        """
        # Clear previous results
        for widget in self.results_frame.winfo_children():
            widget.destroy()
        
        self.result_images = []
        
        query_class = get_class_from_filename(self.search_system.filenames[self.query_idx])
        
        # Display results in grid
        for i, (idx, dist, filename) in enumerate(zip(indices, distances, filenames)):
            row = i // 5
            col = i % 5
            
            # Create frame for each result
            result_frame = ttk.Frame(self.results_frame)
            result_frame.grid(row=row, column=col, padx=5, pady=5)
            
            # Load and display image
            img_path = os.path.join(IMAGES_FOLDER, filename)
            img = Image.open(img_path)
            img.thumbnail((120, 120))
            photo = ImageTk.PhotoImage(img)
            
            img_label = ttk.Label(result_frame, image=photo)
            img_label.pack()
            self.result_images.append(photo)  # Keep reference
            
            # Display info
            result_class = get_class_from_filename(filename)
            is_correct = result_class == query_class
            
            info_text = f"Rank {i+1}\n{filename}\nClass: {result_class}\nDist: {dist:.4f}"
            if is_correct:
                info_text += "\n✓"
            
            info_label = ttk.Label(result_frame, text=info_text, 
                                  foreground='green' if is_correct else 'black')
            info_label.pack()


def main():
    """
    Main function to run the GUI.
    """
    root = tk.Tk()
    app = VisualSearchGUI(root)
    root.mainloop()


if __name__ == '__main__':
    main()


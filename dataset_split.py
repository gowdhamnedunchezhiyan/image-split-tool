        Args:
            images_dir: Directory containing images
            annotations_dir: Directory containing annotations (if separate from images)
            output_dir: Directory where split dataset will be saved
        """
        self.images_dir = images_dir
        self.annotations_dir = annotations_dir or images_dir
        self.output_dir = output_dir
        
    def get_image_annotation_pairs(self):
        """Get pairs of images and their corresponding annotations"""
        # Common image extensions
        # img_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        img_extensions = ['.png']
        # Common annotation extensions
        # ann_extensions = ['.txt', '.xml', '.json']
        ann_extensions = ['.txt']
        
        pairs = []

        print("Images Dir:",self.images_dir)        
        
        # Get all image files
        for ext in img_extensions:
            images = glob.glob(os.path.join(self.images_dir, f"*{ext}"))
            images.extend(glob.glob(os.path.join(self.images_dir, f"*{ext.upper()}")))
        
        for img_path in images:
            img_name = os.path.splitext(os.path.basename(img_path))[0]
            
            # Look for corresponding annotation file
            ann_path = None
            for ann_ext in ann_extensions:
                potential_ann = os.path.join(self.annotations_dir, f"{img_name}{ann_ext}")
                if os.path.exists(potential_ann):
                    ann_path = potential_ann
                    break
            
            if ann_path:
                pairs.append((img_path, ann_path))
            else:
                print(f"Warning: No annotation found for {img_path}")
        
        return pairs
    
    def extract_classes_from_yolo(self, ann_path):
        """Extract class IDs from YOLO format annotation"""
        classes = set()
        try:
            with open(ann_path, 'r') as f:
                for line in f:
                    if line.strip():
                        class_id = int(line.split()[0])
                        classes.add(class_id)
        except:
            pass
        return list(classes)
    
    def extract_classes_from_coco(self, ann_path):
        """Extract class IDs from COCO format annotation"""
        classes = set()
        try:
            with open(ann_path, 'r') as f:
                data = json.load(f)
                if 'annotations' in data:
                    for ann in data['annotations']:
                        classes.add(ann.get('category_id', 0))
        except:
            pass
        return list(classes)
    
    def get_stratification_key(self, pairs, format_type='auto'):
        """Create stratification keys for maintaining class distribution"""
        stratification_keys = []
        
        for img_path, ann_path in pairs:
            if format_type == 'auto':
                # Auto-detect format based on extension
                if ann_path.endswith('.txt'):
                    classes = self.extract_classes_from_yolo(ann_path)
                elif ann_path.endswith('.json'):
                    classes = self.extract_classes_from_coco(ann_path)
                else:
                    classes = [0]  # Default class
            elif format_type == 'yolo':
                classes = self.extract_classes_from_yolo(ann_path)
            elif format_type == 'coco':
                classes = self.extract_classes_from_coco(ann_path)
            else:
                classes = [0]  # Default for unknown formats
            
            # Create a key representing the class combination
            key = tuple(sorted(classes)) if classes else (0,)
            stratification_keys.append(key)
        
        return stratification_keys
    
    def split_dataset(self, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, 
                     random_seed=42, stratify=True, format_type='auto'):
        """
        Split dataset into train/validation/test sets
        
        Args:
            train_ratio: Proportion for training set (default: 0.8)
            val_ratio: Proportion for validation set (default: 0.1)
            test_ratio: Proportion for test set (default: 0.1)
            random_seed: Random seed for reproducibility
            stratify: Whether to maintain class distribution across splits
            format_type: Annotation format ('auto', 'yolo', 'coco', 'none')
        """
        
        # Validate ratios
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError("Ratios must sum to 1.0")
        
        # Set random seed
        random.seed(random_seed)
        
        # Get image-annotation pairs
        pairs = self.get_image_annotation_pairs()
        print(f"Found {len(pairs)} image-annotation pairs")
        
        if len(pairs) == 0:
            raise ValueError("No image-annotation pairs found!")
        
        # Prepare stratification if requested
        stratify_keys = None
        if stratify and len(pairs) > 10:  # Only stratify if we have enough samples
            try:
                stratify_keys = self.get_stratification_key(pairs, format_type)
                print("Using stratified split to maintain class distribution")
            except Exception as e:
                print(f"Stratification failed: {e}. Using random split.")
                stratify_keys = None
        
        # First split: separate test set
        if test_ratio > 0:
            train_val_pairs, test_pairs = train_test_split(
                pairs, 
                test_size=test_ratio, 
                random_state=random_seed,
                stratify=stratify_keys
            )
            
            # Update stratify_keys for remaining data
            if stratify_keys:
                stratify_keys = stratify_keys[:len(train_val_pairs)]
        else:
            train_val_pairs = pairs
            test_pairs = []
        
        # Second split: separate validation from training
        if val_ratio > 0 and len(train_val_pairs) > 1:
            val_size = val_ratio / (train_ratio + val_ratio)
            train_pairs, val_pairs = train_test_split(
                train_val_pairs,
                test_size=val_size,
                random_state=random_seed,
                stratify=stratify_keys if stratify_keys else None
            )
        else:
            train_pairs = train_val_pairs
            val_pairs = []
        
        # Print split statistics
        print(f"\nDataset split:")
        print(f"Training: {len(train_pairs)} samples ({len(train_pairs)/len(pairs)*100:.1f}%)")
        print(f"Validation: {len(val_pairs)} samples ({len(val_pairs)/len(pairs)*100:.1f}%)")
        print(f"Test: {len(test_pairs)} samples ({len(test_pairs)/len(pairs)*100:.1f}%)")
        
        return {
            'train': train_pairs,
            'val': val_pairs,
            'test': test_pairs
        }
    
    def copy_files_to_folders(self, splits, copy_files=True):
        """
        Copy/move files to organized folder structure
        
        Args:
            splits: Dictionary containing train/val/test splits
            copy_files: If True, copy files. If False, create symbolic links (Linux/Mac)
        """
        
        # Create output directory structure
        for split_name in ['train', 'val', 'test']:
            if splits[split_name]:  # Only create if split has data
                os.makedirs(os.path.join(self.output_dir, split_name, 'images'), exist_ok=True)
                os.makedirs(os.path.join(self.output_dir, split_name, 'annotations'), exist_ok=True)
        
        # Copy files
        for split_name, pairs in splits.items():
            if not pairs:
                continue
                
            print(f"\nCopying {split_name} files...")
            
            for i, (img_path, ann_path) in enumerate(pairs):
                # Copy image
                img_dest = os.path.join(self.output_dir, split_name, 'images', 
                                      os.path.basename(img_path))
                # Copy annotation
                ann_dest = os.path.join(self.output_dir, split_name, 'annotations', 
                                      os.path.basename(ann_path))
                
                if copy_files:
                    shutil.copy2(img_path, img_dest)
                    shutil.copy2(ann_path, ann_dest)
                else:
                    # Create symbolic links (Unix systems)
                    try:
                        os.symlink(os.path.abspath(img_path), img_dest)
                        os.symlink(os.path.abspath(ann_path), ann_dest)
                    except:
                        # Fallback to copying if symlink fails
                        shutil.copy2(img_path, img_dest)
                        shutil.copy2(ann_path, ann_dest)
                
                if (i + 1) % 100 == 0:
                    print(f"  Processed {i + 1}/{len(pairs)} files")
        
        print(f"\nDataset split completed! Output saved to: {self.output_dir}")
        
        # Create a summary file
        summary_path = os.path.join(self.output_dir, 'split_summary.txt')
        with open(summary_path, 'w') as f:
            f.write("Dataset Split Summary\n")
            f.write("====================\n\n")
            f.write(f"Total images: {sum(len(pairs) for pairs in splits.values())}\n")
            for split_name, pairs in splits.items():
                f.write(f"{split_name.capitalize()}: {len(pairs)} images\n")
            f.write(f"\nOutput directory: {self.output_dir}\n")


# Example usage
if __name__ == "__main__":
    # Configuration for your YOLO dataset structure
    IMAGES_DIR = "./project-7/images"           # Your images folder
    ANNOTATIONS_DIR = "./project-7/labels"      # Your labels folder  
    OUTPUT_DIR = "./project-7/split_dataset"    # Output directory
    
    # For your 2170 images, using 80/10/10 split
    TRAIN_RATIO = 0.8    # 1736 images
    VAL_RATIO = 0.1      # 217 images  
    TEST_RATIO = 0.1     # 217 images
    
    # Initialize splitter
    splitter = DatasetSplitter(
        images_dir=IMAGES_DIR,
        annotations_dir=ANNOTATIONS_DIR,
        output_dir=OUTPUT_DIR
    )
    
    # Split the dataset
    splits = splitter.split_dataset(
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO, 
        test_ratio=TEST_RATIO,
        random_seed=42,
        stratify=True,  # Maintains class distribution across splits
        format_type='yolo'  # Specify YOLO format for your .txt labels
    )
    
    # Copy files to organized folders
    splitter.copy_files_to_folders(splits, copy_files=True)
    
    # Also copy the classes.txt file to each split
    import shutil
    if os.path.exists("classes.txt"):
        for split_name in ['train', 'val', 'test']:
            if splits[split_name]:  # Only copy if split exists
                shutil.copy2("classes.txt", os.path.join(OUTPUT_DIR, split_name, "classes.txt"))
        print("✅ classes.txt copied to all splits")
    
    print("\n✅ Dataset splitting completed successfully!")
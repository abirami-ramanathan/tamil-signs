from dataset_loader import TLFS23DatasetLoader

dataset_path = r"c:\Users\Abirami Ramanathan\Desktop\tamil-signs\TLFS23 - Tamil Language Finger Spelling Image Dataset"
loader = TLFS23DatasetLoader(dataset_path)
loader.load_dataset_structure()

print("Testing reference images:")
print("-" * 50)

found = 0
test_labels = [0, 13, 31, 50, 100, 150, 200, 246]

for label in test_labels:
    ref = loader.get_reference_image(label)
    class_info = loader.get_class_info(label)
    status = "✓ Found" if ref else "✗ Not found"
    print(f"Label {label:3d} ({class_info['tamil_char']:2s} - {class_info['pronunciation']:5s}): {status}")
    if ref:
        found += 1
        print(f"         Path: {ref}")

print("-" * 50)
print(f"Total found: {found}/{len(test_labels)}")

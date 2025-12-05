import src.utils as utils

# Simulate predict_submission.py logic
TAXONOMY = 'Amazon_products/class_hierarchy.txt'
classes = set()
with open(TAXONOMY, 'r') as f:
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) >= 2:
            parent, child = parts
            classes.add(parent)
            classes.add(child)
        
sorted_classes_pred = sorted(list(classes))
# First 10 classes in prediction logic
print("Prediction Logic (String Sort):", sorted_classes_pred[:10])

# Simulate 4_train_model.py logic
CLASSES_PATH = 'Amazon_products/classes.txt'
id2class, class2id = utils.load_classes(CLASSES_PATH)
# First 10 classes in training logic (assuming 0..N-1)
sorted_ids_train = sorted(list(id2class.keys()))
print("Training Logic (Integer Sort):", sorted_ids_train[:10])

# Check if they match
match = True
for i in range(min(10, len(sorted_classes_pred))):
    if str(sorted_ids_train[i]) != sorted_classes_pred[i]:
        match = False
        break

if not match:
    print("\nMISMATCH DETECTED!")
    print(f"Index 2 in Pred: {sorted_classes_pred[2]}")
    print(f"Index 2 in Train: {sorted_ids_train[2]} (Name: {id2class[sorted_ids_train[2]]})")
else:
    print("\nNo mismatch in first 10 items (unlikely if string vs int sort).")

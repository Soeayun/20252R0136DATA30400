import json

def load_json(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def load_corpus(filepath):
    corpus = {}
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                index = parts[0]
                text = parts[1]
                corpus[index] = text
    return corpus

def load_classes(filepath):
    classes = {}
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                index = int(parts[0])
                name = parts[1]
                classes[index] = name
    return classes

def verify_mapping():
    core_classes_path = '20252R0136DATA30400/checkpoints/core_classes.json'
    corpus_path = '20252R0136DATA30400/Amazon_products/train/train_corpus.txt'
    classes_path = '20252R0136DATA30400/Amazon_products/classes.txt'

    print("Loading files...")
    core_classes = load_json(core_classes_path)
    corpus = load_corpus(corpus_path)
    classes = load_classes(classes_path)

    print(f"Loaded {len(core_classes)} document mappings.")
    print(f"Loaded {len(corpus)} documents.")
    print(f"Loaded {len(classes)} classes.")

    # Check for a few samples
    print("\n--- Sample Verification ---")
    sample_count = 0
    for doc_id, class_ids in core_classes.items():
        if sample_count >= 10:
            break
        
        if doc_id in corpus:
            print(f"\nDocument ID: {doc_id}")
            print(f"Text: {corpus[doc_id][:200]}...") # Print first 200 chars
            
            class_names = []
            for cid in class_ids:
                if cid in classes:
                    class_names.append(f"{classes[cid]} ({cid})")
                else:
                    class_names.append(f"UNKNOWN_CLASS_ID_{cid}")
            
            print(f"Assigned Classes: {', '.join(class_names)}")
        else:
            print(f"\nDocument ID: {doc_id} NOT FOUND in corpus!")
        
        sample_count += 1

    # Check for invalid IDs
    print("\n--- Integrity Check ---")
    invalid_doc_ids = []
    invalid_class_ids = []

    for doc_id, class_ids in core_classes.items():
        if doc_id not in corpus:
            invalid_doc_ids.append(doc_id)
        for cid in class_ids:
            if cid not in classes:
                invalid_class_ids.append(cid)

    if invalid_doc_ids:
        print(f"Found {len(invalid_doc_ids)} invalid document IDs in mapping (e.g., {invalid_doc_ids[:5]})")
    else:
        print("All document IDs in mapping exist in corpus.")

    if invalid_class_ids:
        print(f"Found {len(invalid_class_ids)} invalid class IDs in mapping (e.g., {invalid_class_ids[:5]})")
    else:
        print("All class IDs in mapping exist in classes file.")

if __name__ == "__main__":
    verify_mapping()

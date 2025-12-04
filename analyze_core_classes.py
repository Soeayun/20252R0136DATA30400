import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

def analyze_core_class_distribution(file_path, id2class_path=None):
    """
    Core Class 분포를 심층 분석하고 시각화합니다.
    Args:
        file_path: core_classes.json 파일 경로
        id2class_path: (선택) id2class.txt 파일 경로 (클래스 이름 매핑용)
    """
    print(f"Loading data from: {file_path}")
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"❌ Error loading JSON: {e}")
        return

    # 1. 기본 통계
    doc_ids = sorted([int(k) for k in data.keys()])
    total_docs = len(doc_ids)
    
    # 각 문서당 Core Class 개수 리스트
    counts_per_doc = [len(v) for v in data.values()]
    
    # 모든 Core Class ID 모으기 (Flatten)
    all_classes = []
    for v in data.values():
        all_classes.extend(v)
    
    print("\n" + "="*40)
    print("📊 Core Class Distribution Summary")
    print("="*40)
    print(f"✅ Total Documents: {total_docs}")
    print(f"✅ Total Core Class Instances: {len(all_classes)}")
    print("-" * 30)
    
    if not counts_per_doc:
        print("⚠️ No data found.")
        return

    # 2. 문서당 개수 통계 (Classes per Document)
    print(f"📉 Classes per Document Stats:")
    print(f"   - Min: {min(counts_per_doc)}")
    print(f"   - Max: {max(counts_per_doc)}")
    print(f"   - Mean: {np.mean(counts_per_doc):.2f}")
    print(f"   - Median: {np.median(counts_per_doc):.2f}")
    print(f"   - Std Dev: {np.std(counts_per_doc):.2f}")
    
    # 0개인 문서 체크
    empty_docs = [k for k, v in data.items() if len(v) == 0]
    print(f"   - Empty Docs (0 classes): {len(empty_docs)}")
    if empty_docs:
        print(f"     (Sample IDs: {empty_docs[:5]} ...)")

    # 3. 클래스별 등장 빈도 (Most Common Classes)
    class_counts = Counter(all_classes)
    most_common = class_counts.most_common(20)
    
    # 클래스 이름 매핑 (id2class 파일이 있다면)
    id2name = {}
    if id2class_path:
        try:
            with open(id2class_path, 'r') as f:
                for line in f:
                    parts = line.strip().split('\t') # 탭이나 공백 구분 확인 필요
                    if len(parts) >= 2:
                        id2name[int(parts[0])] = parts[1]
        except:
            print("⚠️ id2class mapping skipped (file not found or format error).")

    print("\n🏆 Top 20 Most Frequent Core Classes:")
    print(f"{'Class ID':<10} {'Freq':<10} {'Name (if available)':<20}")
    print("-" * 40)
    for cid, freq in most_common:
        name = id2name.get(cid, "Unknown")
        print(f"{cid:<10} {freq:<10} {name:<20}")

    # 4. 시각화 (Visualization)
    plt.figure(figsize=(15, 5))

    # A. 문서당 클래스 개수 분포 (Histogram)
    plt.subplot(1, 2, 1)
    sns.histplot(counts_per_doc, bins=range(min(counts_per_doc), max(counts_per_doc) + 2), kde=True, color='skyblue')
    plt.title('Distribution of # Core Classes per Document')
    plt.xlabel('Number of Core Classes')
    plt.ylabel('Document Count')
    plt.axvline(np.mean(counts_per_doc), color='r', linestyle='--', label=f'Mean: {np.mean(counts_per_doc):.1f}')
    plt.legend()

    # B. 클래스별 등장 빈도 분포 (Long-tail 확인용)
    plt.subplot(1, 2, 2)
    sorted_counts = sorted(class_counts.values(), reverse=True)
    plt.plot(sorted_counts, color='orange')
    plt.title('Class Frequency Distribution (Long-tail Check)')
    plt.xlabel('Class Rank')
    plt.ylabel('Frequency')
    plt.yscale('log') # 로그 스케일로 보면 Long-tail이 잘 보임
    plt.grid(True, which="both", ls="-", alpha=0.2)

    plt.tight_layout()
    plt.show()

# --- 실행 ---
# 경로를 본인 환경에 맞게 수정하세요.
core_file = 'checkpoints/core_classes.json'
# id2class 파일이 있다면 경로를 넣어주세요 (없으면 None)
id_map_file = None 

analyze_core_class_distribution(core_file, id_map_file)
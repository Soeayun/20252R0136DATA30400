"""
Step 2: Generate synthetic reviews using GPT-4o-mini

Generates high-quality, class-specific synthetic reviews using real examples as templates.

Requires: OPENAI_API_KEY in .env file or environment variable
Output: synthetic_reviews.jsonl
"""

import json
import os
import time
import random
import sys
sys.path.append('..')
from openai import OpenAI
from tqdm import tqdm
from src import utils
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed

# Load environment variables from .env file
load_dotenv()

# API Configuration
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise ValueError("OPENAI_API_KEY not found. Please set it in .env file or environment variable")

client = OpenAI(api_key=API_KEY)

# Generation parameters
REVIEWS_PER_CALL = 15
REVIEW_LENGTH = "70-100 words"
TEMPERATURE = 0.9
MAX_TOKENS = 4000  # Increased for 30 reviews (Level 2 classes)
MAX_API_CALLS = 1000  # Hard limit

SYSTEM_PROMPT = """You are an Amazon customer writing authentic product reviews.
Write exactly like real customers do - natural, conversational, with personal experiences and specific details."""


def load_real_review_examples(corpus, num_samples=5):
    """Load random real reviews as examples"""
    sample_ids = random.sample(list(corpus.keys()), min(num_samples, len(corpus)))
    return [corpus[sid] for sid in sample_ids]


def build_prompt(category, num_reviews, real_examples):
    """
    Build prompt for generating N reviews for a SINGLE category
    
    Args:
        category: Single category dict
        num_reviews: Number of reviews to generate
        real_examples: List of real review texts
    Returns:
        Prompt string
    """
    # Start with real examples
    prompt = "Write product reviews that sound EXACTLY like these real Amazon customer reviews:\n\n"
    prompt += "--- REAL EXAMPLES (COPY THIS STYLE) ---\n"
    
    for idx, example in enumerate(real_examples[:3], 1):
        truncated = ' '.join(example.split()[:100])
        prompt += f"\nExample {idx}:\n\"{truncated}\"\n"
    
    prompt += "\n--- YOUR TASK ---\n"
    prompt += f"Write {num_reviews} reviews for the following product category:\n\n"
    
    prompt += f"Category: {category['name']} (Parent: {category['parent_name']})\n"
    prompt += f"Keywords: {', '.join(category['keywords'][:4])}\n"
    prompt += f"→ Mention specific features that make this different from general \"{category['parent_name']}\" products\n\n"
    
    prompt += "CRITICAL RULES:\n"
    prompt += "1. LENGTH: Vary from 40-120 words (don't make them all the same)\n"
    prompt += "2. TONE: Mix it up - 40% positive, 40% mixed feelings, 20% critical\n"
    prompt += "3. BE NATURAL: Write like you're texting a friend, not writing an essay\n"
    prompt += "4. SHOW REALITY: Include pros AND cons (\"works great but...\", \"love X, wish Y was better\")\n"
    prompt += "5. BE SPECIFIC: Use real numbers (\"3 months\", \"2 year old\", \"about $30\")\n"
    prompt += "6. AVOID PATTERNS: Don't start every review the same way\n\n"
    
    prompt += f"""
Output ONLY a JSON array of {num_reviews} review strings (no category_id needed):
["review text 1", "review text 2", ...]

WRITING STYLE:
✓ Write conversationally (like talking to a friend)
✓ Vary length (40-120 words, don't make them all the same)
✓ Mix emotions (most things have pros AND cons)
✓ Be specific with details/numbers
✓ Sound like a real person, not a marketing copy
✗ Don't make everything perfect
✗ Don't start every review with "I bought"
✗ Don't overuse "game changer", "worth it", "highly recommend"
"""
    
    return prompt


def call_openai(category_id, prompt, retry=3):
    """
    Call OpenAI API with retry logic
    
    Args:
        category_id: Category ID to attach to reviews
        prompt: Prompt string
        retry: Number of retries
    
    Returns:
        (reviews, tokens) where reviews = [{"category_id": X, "review": "..."}]
    """
    for attempt in range(retry):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                response_format={"type": "json_object"}
            )
            
            # Parse response
            content = response.choices[0].message.content
            
            # Try to extract JSON
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            
            data = json.loads(content)
            
            # Handle different formats
            if isinstance(data, dict):
                if "reviews" in data:
                    review_texts = data["reviews"]
                elif "array" in data:
                    review_texts = data["array"]
                else:
                    # Might be wrapped in a key
                    review_texts = list(data.values())[0] if data else []
            elif isinstance(data, list):
                review_texts = data
            else:
                return None, 0
            
            # Convert to proper format with category_id
            reviews = [
                {"category_id": category_id, "review": text}
                for text in review_texts
                if isinstance(text, str) and len(text) > 10
            ]
            
            return reviews, response.usage.total_tokens
            
        except Exception as e:
            print(f"  Attempt {attempt + 1} failed: {e}")
            if attempt < retry - 1:
                time.sleep(2 ** attempt)
            else:
                return None, 0
    
    return None, 0


def generate_reviews():
    """Main generation loop with parallel processing"""
    # Get script directory for correct paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    data_dir = os.path.join(project_dir, 'Amazon_products')
    
    # Load train corpus for real examples
    print("Loading train corpus...")
    train_corpus = utils.load_corpus(os.path.join(data_dir, 'train', 'train_corpus.txt'))
    print(f"Loaded {len(train_corpus)} real reviews\n")
    
    # Load selected classes
    print("Loading selected classes...")
    with open(os.path.join(script_dir, 'selected_classes.json'), 'r') as f:
        data = json.load(f)
    
    classes = data['classes']
    total_classes = len(classes)
    
    print(f"Loaded {total_classes} classes")
    print(f"Target reviews: {data['total_reviews']}")
    print(f"Estimated API calls: ~{data['total_reviews'] // REVIEWS_PER_CALL}")
    print(f"Max API calls limit: {MAX_API_CALLS}\n")
    
    # Prepare batches: One category per batch
    # Each batch = (category, num_reviews_to_generate)
    batches = []
    
    for cls in classes:
        if cls['num_reviews'] > 0:
            batches.append((cls, cls['num_reviews']))
    
    # Enforce MAX_API_CALLS limit
    if len(batches) > MAX_API_CALLS:
        print(f"WARNING: {len(batches)} batches exceeds limit. Truncating to {MAX_API_CALLS}")
        batches = batches[:MAX_API_CALLS]
    
    total_expected = sum(num for _, num in batches)
    print(f"Created {len(batches)} batches (≤ {MAX_API_CALLS} limit)")
    print(f"Expected total reviews: {total_expected}\n")
    
    # Process function for each batch
    def process_batch(batch_data):
        idx, (category, num_reviews) = batch_data
        
        # Sample real examples
        real_examples = load_real_review_examples(train_corpus, num_samples=5)
        
        # Build prompt for this category
        prompt = build_prompt(category, num_reviews, real_examples)
        
        # Call API
        reviews, tokens = call_openai(category['category_id'], prompt)
        
        return idx, reviews, tokens
    
    # Parallel execution
    print("Generating reviews in parallel (10 workers)...\n")
    all_reviews = []
    total_tokens = 0
    failed_batches = []
    max_workers = 10
    
    output_file = open(os.path.join(script_dir, 'synthetic_reviews.jsonl'), 'w')
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        futures = {
            executor.submit(process_batch, (i, batch)): i
            for i, batch in enumerate(batches)
        }
        
        # Process results as they complete
        with tqdm(total=len(batches), desc="Generating") as pbar:
            for future in as_completed(futures):
                batch_idx, reviews, tokens = future.result()
                
                if reviews is None:
                    failed_batches.append(batch_idx)
                    pbar.update(1)
                    continue
                
                total_tokens += tokens
                
                # Write to file
                for review in reviews:
                    output_file.write(json.dumps(review) + '\n')
                    all_reviews.append(review)
                
                pbar.update(1)
    
    output_file.close()
    
    # Summary
    print(f"\n{'='*60}")
    print(f"Generation Complete!")
    print(f"{'='*60}")
    print(f"Total reviews generated: {len(all_reviews)}")
    print(f"Total API calls: {len(batches) - len(failed_batches)}")
    print(f"Failed batches: {len(failed_batches)}")
    print(f"Total tokens used: {total_tokens:,}")
    print(f"Estimated cost: ${total_tokens * 0.00000015:.2f}")
    print(f"\nOutput saved to: synthetic_reviews.jsonl")
    
    # Save metadata
    metadata = {
        "total_reviews": len(all_reviews),
        "total_calls": len(batches) - len(failed_batches),
        "failed_calls": len(failed_batches),
        "total_tokens": total_tokens,
        "reviews_per_call": REVIEWS_PER_CALL,
        "temperature": TEMPERATURE,
        "max_calls_limit": MAX_API_CALLS,
        "used_real_examples": True,
        "parallel_workers": max_workers
    }
    
    with open(os.path.join(script_dir, 'generation_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)


if __name__ == "__main__":
    generate_reviews()

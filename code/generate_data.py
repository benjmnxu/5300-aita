import pandas as pd
from openai import AsyncOpenAI
import asyncio
import sys
from enum import Enum
from synth_prompt import synthetic_prompt
import random
from datetime import datetime, timedelta

mapping = {
    "YTA": "You're the Asshole",
    "ESH": "Everyone sucks here",
    "NAH": "No one is the Asshole",
    "NTA": "Not the Asshole",
}

dev_path = "data/dev.csv"
train_path = "data/train.csv"
test_path = "data/test.csv"
output_base = "data/synthetic_{target}_1800.csv"

dev_data = pd.read_csv(dev_path)
train_data = pd.read_csv(train_path)
test_data = pd.read_csv(test_path)

data = pd.concat([dev_data, train_data, test_data], axis=0, ignore_index=True)

client = AsyncOpenAI(api_key="API_KEY")

semaphore = asyncio.Semaphore(50)

def random_date(start_date, end_date):
    """Generate a random date between start_date and end_date."""
    delta = end_date - start_date
    random_days = random.randint(0, delta.days)
    return start_date + timedelta(days=random_days)

def get_random_example(filtered_data):
    """
    Select a few random examples for use as demonstration prompts.
    """
    return filtered_data['body'].sample(1).iloc[0]


# Generate synthetic summary
async def generate_synthetic_example(target_class, example):
    global iteration
    async with semaphore:
        try:
            response = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": synthetic_prompt(target_class, example)}
                ],
                temperature=0.8,
            )

            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error generating text: {e}")
            return None

# Generate synthetic data
async def generate_synthetic_data(filtered_data, target_class, num_samples):
    """
    Generate synthetic data by providing sampled examples to the LLM.
    """
    print(f"Generating {num_samples} synthetic examples for '{target_class}'. This may take a while...")

    # Start synthetic generation tasks
    tasks = [
        asyncio.create_task(generate_synthetic_example(target_class, get_random_example(filtered_data)))
        for _ in range(num_samples)
    ]

    completed_data = []
    failed_count = 0

    # Wait for tasks with timeout handling
    while tasks:
        done, pending = await asyncio.wait(tasks, timeout=20, return_when=asyncio.FIRST_COMPLETED)

        for task in done:
            try:
                result = task.result()
                if result:
                    completed_data.append(result)
                else:
                    failed_count += 1
            except Exception as e:
                print(f"Task failed: {e}")
                failed_count += 1

        # Remove completed tasks from the list
        tasks = list(pending)

        # Progress logging
        print(f"Progress: {len(completed_data)} completed, {failed_count} failed, {len(tasks)} pending...")

        # Optional: Cancel pending tasks if too many failures occur
        if failed_count > num_samples * 0.2:  # Stop if more than 20% tasks fail
            print("Too many failures, cancelling remaining tasks...")
            for task in pending:
                task.cancel()
            break

    print(f"Completed generation. Success: {len(completed_data)}, Failed: {failed_count}")
    return completed_data


# Main function
async def main(target_class, num_samples):
    filtered_data = data[data['target'] == target_class]  # Filter data for the target class

    # Generate synthetic examples
    synthetic_posts = await generate_synthetic_data(filtered_data, target_class, num_samples)
    # Generate random metadata
    start_date = datetime(2018, 1, 1)
    end_date = datetime.now()
    metadata = []
    for body in synthetic_posts:
        metadata.append({
            "verdict": mapping[target_class],
            "body": body,
            "num_comments": random.randint(0, 1000),
            "score": random.randint(500, 3000),
            "upvote_ratio": round(random.uniform(0.5, 1.0), 2),
            "title": "AITA RANDOM TITLE",
            "id": f"post_{random.randint(10000, 99999)}",
            "date": random_date(start_date, end_date).strftime("%Y-%m-%d %H:%M:%S"),
            "target": target_class,
            "length_body": len(body),
            "title_aita": "AITA"
        })
            
    output_path = output_base.format(target=target_class)
    synthetic_df = pd.DataFrame(metadata)
    synthetic_df.to_csv(output_path, index=False)
    print(f"Saved {num_samples} synthetic '{target_class}' examples to {output_path}")

if __name__ == "__main__":
    # Check the number of arguments
    if len(sys.argv) != 3:
        print(f"Usage: python {sys.argv[0]} <Class name (YTA, ESH, NAH, NTA)> <num samples>")
        sys.exit(1)

    target_class = sys.argv[1].upper()
    try:
        num_samples = int(sys.argv[2])
        if num_samples <= 0:
            raise ValueError("Number of samples must be a positive integer.")
    except ValueError as e:
        print(f"Invalid number of samples: {e}")
        sys.exit(1)

    if target_class not in mapping:
        print(f"Invalid target class. Must be one of {list(mapping.keys())}")
        sys.exit(1)

    print(f"Target class: {target_class}")
    print(f"Number of samples: {num_samples}")

    asyncio.run(main(target_class, num_samples))

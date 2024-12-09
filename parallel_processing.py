import openai
import pandas as pd
from threading import Thread, Lock
import time

openai.api_key = "OPENAI_API_KEY"

csv_path = "data/balanced_train_dataset.csv"
data = pd.read_csv(csv_path)

test_data = data.iloc[2000:4000].copy()

summaries = [None] * len(test_data)
lock = Lock() 

def generate_summary(text):
    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"Summarize the following text in a single 2-3 sentence paragraph:\n{text}"}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error generating summary for text: {text[:50]}...: {e}")
        return None

def process_row(index, body_text):
    summary = generate_summary(body_text)
    with lock:
        summaries[index] = summary



threads = []
for index, body_text in enumerate(test_data['body']):
    thread = Thread(target=process_row, args=(index, body_text))
    threads.append(thread)
    thread.start()


for thread in threads:
    thread.join()

test_data['body_summary'] = summaries

output_path = "data/balanced_train_dataset_augmented.csv"
test_data.to_csv(output_path, index=False)
print(f"Saved augmented data to {output_path}")

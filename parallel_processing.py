import pandas as pd
import openai
import asyncio
from aiohttp import ClientSession

openai.api_key = "key"

csv_path = "data/dev.csv"
already_augmented_path = "data/dev.csv"
output_path = "data/predicted_dev.csv"

data = pd.read_csv(csv_path)
test_data = data.iloc[1501:2250].copy()
# already_augmented_data = pd.read_csv(already_augmented_path)

# Remove already augmented rows
# test_data = data[~data['id'].isin(already_augmented_data['id'])].reset_index(drop=True)


summaries = [None] * len(test_data)
# lock = Lock()

async def generate_summary(text):
    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role":"user", "content": f"You are a classifier for posts in the Am I the Asshole subreddit. Your task is to classify each post into one of the following categories: \"NTA\" (Not the Asshole), \"YTA\" (You're the Asshole), \"ESH\" (Everyone's the Asshole), and \"NAH\" (No Assholes Here), primarily based on the interpersonal behavior the user describes in the body text. Provide only the category name as the output. :\n{text}. Category: "}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error generating summary for text: {text[:50]}...: {e}")
        return None

async def process_data(test_data):
    summaries = []
    tasks = []
    for body_text in test_data['body']:
        tasks.append(generate_summary(body_text))

    # Gather results with asyncio
    summaries = await asyncio.gather(*tasks)
    return summaries

async def main():
    summaries = await process_data(test_data)
    test_data['target'] = summaries
    test_data.to_csv(output_path, index=False)
    print(f"Saved augmented data to {output_path}")

# Run the asyncio event loop
asyncio.run(main())

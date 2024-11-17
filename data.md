# Data Analysis - r/AITA Analysis


For this experiment, we decided to primarily use the dataset collected by [@albechen on GitHub](https://github.com/albechen/aita-nlp-classification), as we believed that it had the best mixture of included metadata (including most features we deemed to be important about the post), documentation/code of scraping methods, and metric analysis. While other online datasets might have included more features, we believed this dataset’s documentation and reproducibility for its scraping method gave us the most flexibility in adding to/expanding (scraping newer data) and understanding the dataset as a whole. Additionally, this GitHub repository included an example of a basic CNN model trained on the same dataset (produced several years ago), which we figured could serve as a good starting point or baseline testing model for our explorations moving forward.

The dataset contains posts from the subreddit **"Am I the Asshole" (AITA)**, focusing on individual posts (title, body, metadata) and the sentiment expressed in verdicts (comments and aggregated comments written by the community). These verdicts primarily include **"NTA" (Not the Asshole)** and **"YTA" (You're the Asshole)**, along with other classifications like **"ESH" (Everyone's the Asshole)** and **"NAH" (No Assholes Here)**. After parsing, the dataset contains the following structure:

### Columns

- **index**: Index column
- **verdict**: The outcome based on user comments (e.g., "Not the A-hole," "Asshole")
- **body**: The main content of the post describing the situation
- **num_comments**: Number of comments on the post
- **score**: The post's score (total upvotes)
- **upvote_ratio**: The ratio of upvotes to downvotes
- **title**: Title of the post
- **id**: Reddit post ID
- **date**: Timestamp of the post
- **target**: Simplified verdict collected from user comments as judged by the community (e.g., "NTA" for Not the Asshole)
- **length_body**: Length of the post's body content
- **title_aita**: Type of post, either "AITA" or "WIBTA"

### Example Entry

An example row in the dataset includes:

- **verdict**: No A-holes here
- **body**: Let me preface this off by blaming all of you, the entirety of reddit for desensitizing me, and giving me major trust issues on April fools day. So here it goes... Yesterday I got a group text from my wife’s (Sarah) side of the family stating there was an emergency family meeting happening that night over dinner at my mother-in-law’s (Barb) house. I immediately had April fools spidey senses starting to tingle, but we haven't all got together since Christmas so I overlooked it and said we (my wife and I) were in. We were the last to arrive and it was pretty somber when we walked in. We all sat down at the table and my wife’s brother (Tim) informed the family that his wife (Ashley) has been having an affair and they are divorcing. The affair was with a longtime close family friend (Chris) who lived a block away. Chris' wife (Jen) had caught them when she came home early one day last week and broke the news to my brother-in-law Tim. Both families have been friends for years. They live less than a block from each other, each have been married for 15+ years, and have 4 kids around the same age. Honestly, I always thought both families were picture-perfect. Hell, all four of them and their kids were at our house two weeks ago for a BBQ. After airing a lot of dirty laundry, discussing plans to divorce, and how it could affect future family functions, Tim opened it up to the group for questions... there was silence. I broke the silence with laughter and a slow clap, saying this was the best April fools gag I'd ever seen but that I wasn’t falling for it. I told Ashley and especially Tim they need to consider going into theater, as their performances were top-notch and the tears seemed genuine. Being the newest family member (my wife and I married six months ago), this was probably not the best thing to say in hindsight. I probably should not have said anything. Everyone in the room looked horrified. My mother-in-law, who had been crying the entire time, lost all composure, left the room in hysterics, and did not return before we left. Tim just shook his head, and his cheating wife actually let out a brief chuckle before calling me a dumbass for thinking this was a ruse, and then berated me for being insensitive. The rest of the family sat in silence, shaking their heads as my wife berated me for trying to make a joke out of a serious situation... I am still dumbfounded. In hindsight, I probably should have sat in silence... but I honestly still feel like I was calling out an April fools gag. Am I the asshole?
- **num_comments**: 1815
- **score**: 19163
- **upvote_ratio**: 0.88
- **title**: AITA for thinking a divorce announcement due to an affair was an April fools joke?
- **id**: b8m3yb
- **date**: 2019-04-02 17:19:23
- **target**: NAH
- **length_body**: 2614
- **title_aita**: AITA


### Dataset Linking/Splitting
We wrote and used the following code to create a train/dev/test split of the full dataset. The gzipped tar archive file containing all three files can be found at https://tinyurl.com/cis5300-aitadata, and was submitted on Gradescope along with this file.

```python
import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv("/content/aita_cleaned_full.csv")
train_data, temp_data = train_test_split(data, test_size=0.2, random_state=42)

# 80/10/10 train/dev/test split
dev_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

train_data.to_csv("aita_data/train/train.csv", index=False)
dev_data.to_csv("aita_data/dev/dev.csv", index=False)
test_data.to_csv("aita_data/test/test.csv", index=False)
```


def synthetic_prompt(category: str, example: str):
    return f"""
    
    You are writing a post in the Am I the Asshole subreddit. Based primarily on your interpersonal behavior, the subreddit will classify your story into one of the following categories: \"NTA\" (Not the Asshole), \"YTA\" (You're the Asshole), \"ESH\" (Everyone's the Asshole), and \"NAH\" (No Assholes Here). 
    Your goal is to write a realistic story that will result in your post being classified as {category}.

    Here is an example of a {category} story. ONLY USE THE EXAMPLE TO GET AN IDEA, BUT YOU MUST USE A DIFFERENT STORY:
        {example}

    OUTPUT ONLY YOUR STORY:
    """
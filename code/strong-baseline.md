## Strong Baseline Outline

The code expects the following files (which are all in our Github repo):

```train.csv```: Training data.
```dev.csv```: Development (validation) data.

Each file should contain the following columns:

```body```: Text content (e.g., the main article or post body).
```title```: Title of the article or post.
```verdict```: Target classification labels.

To run this script, simply run ```python strong_baseline.py```. We recommend this script be run on a GPU (we originally ran it as a cell in a Google Colab notebook using the T4 GPU runtime environment). This will also write the predicted labels to a file called ```predicted_labels_strong.txt``` which can be used in conjunction with ```score.py``` to provide more granular accuracy metrics (though the overall accuracy is printed when the script is run). 

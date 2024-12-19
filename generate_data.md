generate_data.py is how we generated 5600 synthetic datapoints (1800 for YTA, NAH, and ESH). The command line interface is simply:

`python generate_data.py <CLASS> <NUMBER OF ENTRIES TO GENERATE>`

For example, if I wanted to generate 1800 YTA entries: `python generate_data.py YTA 1800`

It will write results to the file ``data/synthetic_{target}_1800.csv``, where the target is one of the aforementioned classes (NTA, YTA, NAH, ESH). Any previous data will be overwritten. It took me around 5-10 minutes and 40 cents to generate 1800 entries. The file is verbose: progress will be continuously printed out onto terminal.

To run, please make sure you are using your api key on line 28.
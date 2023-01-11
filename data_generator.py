import requests
import pandas as pd


# Download data from internet (direct link to csv)
url = 'https://gist.githubusercontent.com/omarish/5687264/raw/7e5c814ce6ef33e25d5259c1fe79463c190800d9/mpg.csv'
file = requests.get(url)
with open ('csv_file','wb') as f:
    f.write(file.content)


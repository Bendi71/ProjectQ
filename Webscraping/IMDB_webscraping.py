import re

from urllib.request import urlopen
from bs4 import BeautifulSoup

from datetime import datetime
import pandas as pd


url = "https://www.imdb.com/search/title/?count=100&groups=top_1000&sort=user_rating"
html_page = urlopen(url)
soup = BeautifulSoup(html_page, "html.parser")



s_title = soup.findAll('h3', class_='lister-item-header')
s_rating=soup.findAll('div', class_='ratings-bar')
titles = []
for cimek in s_title:
      ss_title=cimek.a.text
      titles.append(ss_title)
      #print(ss_title)
ratings = []
for ertek in s_rating:
      ss_rating=ertek.strong.text
      ratings.append(ss_rating)
      #print(ss_rating)

for i in range(len(titles)):
      print(f'Cím: {titles[i]}, értékelés: {ratings[i]}')
#print(titles,ratings)
from bs4 import BeautifulSoup
import requests
import time

def find_jobs():
    html_text=requests.get('https://www.profession.hu/allasok/1,0,0,python%401%401?keywordsearch').text
    soup=BeautifulSoup(html_text,'lxml')
    jobs=soup.find_all('div', class_='card-body')

    for index, job in enumerate(jobs):
        company=job.find('div',class_='card-body-header')
        other=job.find('div',class_='position_and_company')
        pub_date = other.find('div', class_='text-right w-100').text
        if 'ma' or 'Friss' in pub_date:
            company_name=company.find('a',class_='link-icon').text
            place=company.find('div',class_='job-card__company-address newarea mt-2 mt-md-0 icon map').text
            more_info=company.h2.a['href']
            with open(f'posts/{index}.txt','w') as f:
                f.write(f'Név: {company_name.strip()}, Hely: {place.strip()}, Hirdetés feladásának ideje: {pub_date.strip()} \n')
                f.write(f' További információ: {more_info} \n')
                f.write('\n')
            print(f'File saved: {index}')

if __name__ == '__main__':
    while True:
        find_jobs()
        time_wait=1
        print(f'Waiting {time_wait} minutes...')
        time.sleep(time_wait*60)
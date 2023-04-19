#!/usr/bin/env python
# coding: utf-8

# # SimplyHired Scraping Script 
# ##### Dated on: 19 - 04 - 2023

# ### Importing libraries

# In[4]:


from bs4 import BeautifulSoup
import requests
import pandas as pd


# ### Main script

# In[ ]:


# Main page soup object
url = 'https://www.simplyhired.co.in/search?q=data+scientist&l=India'
response = requests.get(url)
soup = BeautifulSoup(response.content, 'lxml')

# Initialize
data = []
n = 0

# Iterate over pages
while n < 100:
    # Move to the next page
    try:
        next_page = soup.find('a', class_ = 'next-pagination').get('href')
    except AttributeError:
        pass
        
    baselink = 'https://www.simplyhired.co.in'
    compound_link = baselink + next_page
    print(f'Page {n}:', compound_link)

    # Empty containers
    Url = []
    Title = []  
    list_ = []
    list_1 = []
    Company = []
    Location = []
    Working_time = []
    
    # Fetch all the details of a job post
    job_container = soup.find_all(class_ = 'jobposting-title-container')
    for i in job_container:
        embedded_link = i.find('a')['href']
        page_link = baselink + embedded_link
        Url = page_link
        
        response_embedded = requests.get(page_link)
        soup_embedded = BeautifulSoup(response_embedded.content, 'lxml')
    
        # Job title
        try:
            Title = soup_embedded.find(class_ = 'viewjob-jobTitle h2').string
                
        except AttributeError:
            Title = ''
            
        # Job location
        try:
            info = soup_embedded.find(class_ = 'viewjob-header-companyInfo').get_text()
            list_ = info.split(' - ')
        
            Company = list_[0]
            Location = list_[1]
                
        except AttributeError:
            Company = ''
            Location = ''
            
        # Job qualification        
        try:
            soup_embedded.find('div', {'class': 'viewjob-qualifications'})
            qualification_section = soup_embedded.find('div', {'class': 'viewjob-qualifications'})
            qualification_list = qualification_section.find('ul', {'class': 'Chips'})
            list_1 = [li.text for li in qualification_list.find_all('li')]
        
        except AttributeError:
            list_1 = []
            
        data.append([Title, Company, Location, list_1, Url])
        
    # Update soup object
    response = requests.get(compound_link)
    soup = BeautifulSoup(response.content, 'lxml')
    
    n += 1
    
# Save dataframe to csv
df = pd.DataFrame(data, columns=['Title', 'Company', 'Location', 'Qualifications', 'Url'])
df.to_csv('sample.csv', index = True)


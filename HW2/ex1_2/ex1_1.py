import requests
import time
from lxml import html
from bs4 import BeautifulSoup
from tqdm import tqdm
import csv

def get_amazon_products(num_pages, keyword):
    # I generated with chatgpt some useful headers to seem like a browser in the requests
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.0.0 Safari/537.36",
        "Connection": "keep-alive",
    }
    
    all_products = set()
    repetitions = 0
    for i in range(num_pages):
        print(f"Downloading page {i+1}")
        # Define the URL and request the page
        url = f"https://www.amazon.it/s?k={keyword}&page={i+1}"
        res = requests.get(url, headers=headers)
        # launches an exception if the request wasn't succesful
        res.raise_for_status()
        

        # initialize the BeautifulSoup object with the html parser for the response 
        soup = BeautifulSoup(res.content, "html.parser")

        # I identified on the browser the container of the search results
        search_results = soup.find("div", {"class": "s-main-slot s-result-list s-search-results sg-row"})

        # I found also the containers of the products and I isolated their content 
        products = search_results.find_all("div", {"class":"template=SEARCH_RESULTS"})

        # during debugging I checked a bit the html structure by storing it in a file
        #with open("search_results.html", "w", encoding="utf-8") as f:
        #    f.write(str(search_results))
        last_iter = all_products.copy()
        for p in products:
            # I identified the HTML elements with all the needed information. stars or price info may be missing and such tuples were discarded
            desc = p.find("span", {"class": "a-size-base-plus a-color-base a-text-normal"}).get_text()
            
            # I added the amazon prefix to the href content, composing working links
            #link = "https://www.amazon.it" + p.find("a", {"class": "a-link-normal s-underline-text s-underline-link-text s-link-style a-text-normal"})["href"]
            # the links provided by amazon are pre-elaborated and contain duplicates: same product with different links
            # to obtain unique urls, we recover the asin (10-characters id code) of the items
            asin = p["data-csa-c-item-id"][-10:]
            link = "https://www.amazon.it/dp/" + asin

            price = p.find("span", {"class":"a-offscreen"})
            # very few products don't have the price specified
            if price is None:
                continue
            else:
                # preprocess the price to make it a float from "1.342,99 $" to 1342.99 
                price = float(price.get_text().split()[0].replace(".", "").replace(",","."))

            is_prime = p.find("i", {"aria-label":"Amazon Prime"}) is not None 

            stars = p.find("span", {"class":"a-icon-alt"})
            if stars is None:
                continue
            else:
                # preprocess also the stars from "4,2 stelle su 5" to "4.2"
                stars = float(stars.get_text()[:3].replace(",","."))

            #preprocess desc
            all_products.add((desc, price, is_prime, link, stars))

        #check that the new page contains at least one new product
        if len(last_iter) == len(all_products):
            #print("Repeating at page:", i+1)
            repetitions +=1
            # stop downloading: 10 times same products in a row
            if repetitions >= 10:
                break
        else:
            repetitions = 0

        time.sleep(0.5)

    return all_products


def save_products_to_tsv(products, file="products.tsv"):
    #newline parameter avoids double \n
    with open(file, mode='w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        
        # Write the header row (column names)
        writer.writerow(['Description', 'Price', 'Prime', 'URL', 'Number of stars'])
        
        # Write the data rows
        for product in products:
            writer.writerow(product)

N = 100
KEYWORD = "computer"

prods = get_amazon_products(N, KEYWORD)
save_products_to_tsv(prods)

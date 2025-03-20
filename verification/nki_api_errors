import requests
from bs4 import BeautifulSoup

# URL of the webpage you want to parse
url = 'https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/nki/api/nki.errors.html#nki-api-errors'  # Replace with your target URL

# Send a GET request to fetch the webpage
response = requests.get(url)

# Check if the request was successful (status code 200)
if response.status_code == 200:
    # Parse the content of the page using BeautifulSoup
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Get the textual content from the parsed HTML (you can refine this as needed)
    page_text = soup.get_text()

    # Write the content to a text file
    with open('nki_api_errors.txt', 'w', encoding='utf-8') as file:
        file.write(page_text)

    print("Page content saved to output.txt")
else:
    print(f"Failed to retrieve the webpage. Status code: {response.status_code}")

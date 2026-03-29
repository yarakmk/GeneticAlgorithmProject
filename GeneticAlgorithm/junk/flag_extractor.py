import requests
from bs4 import BeautifulSoup
import re
import json

def scrape_gcc_flags():
    """
    Scrapes GCC optimization flags, specifically handling the 'no-' prefix 
    to extract the base optimization names as mentioned in the documentation.
    """
    url = "https://gcc.gnu.org/onlinedocs/gcc/Optimize-Options.html"
    print(f"Fetching documentation from {url}...")
    
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
    except Exception as e:
        print(f"Failed to fetch page: {e}")
        return

    soup = BeautifulSoup(response.text, 'html.parser')
    flags = []
    
    # Blacklist of keywords for safety and relevance
    blacklist = [
        "debug", "stack", "warn", "instrument", "sanitize", "profile", 
        "param", "live-patching", "tracking", "delayed-branch"
    ]

    for code in soup.find_all('code'):
        text = code.get_text().strip()
        
        # We look for anything starting with -f
        if text.startswith('-f'):
            # Skip flags with '=' (parameters)
            if '=' in text:
                continue
                
            # Regex to ensure it's a valid flag format
            if re.match(r'^-f[a-z0-9\-]+$', text):
                flag_name = text[2:] # Strip '-f'
                
                # If the documentation lists the negative version '-fno-optimization'
                # we strip the 'no-' to get the base optimization name 'optimization'
                if flag_name.startswith('no-'):
                    flag_name = flag_name[3:]
                
                # Check against the blacklist
                if any(word in flag_name for word in blacklist):
                    continue
                
                # Add to list if unique
                if flag_name not in flags:
                    flags.append(flag_name)

    print(f"Successfully extracted {len(flags)} candidate base flags.")
    
    # Save the base names for the GA to toggle
    with open("all_flags.txt", "w") as f:
        for flag in flags:
            f.write(f"{flag}\n")
    
    with open("all_flags.json", "w") as f:
        json.dump(flags, f, indent=4)

    print("Files 'all_flags.txt' and 'all_flags.json' have been updated.")

if __name__ == "__main__":
    scrape_gcc_flags()
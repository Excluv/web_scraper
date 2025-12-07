from urllib.parse import urlparse

from logger import ScraperLogger


logger = ScraperLogger(__name__).logger

def get_domain_name(url: str):
    try:
        parsed_url = urlparse(url)
        return parsed_url.domain
    except Exception as err:
        logger.error(f"Failed to retrieve domain name from {url}", exc_info=True)
        raise


def process_hyper_links(entry_url: str, hyper_links: list):
    domain = get_domain_name(entry_url)
    for link in hyper_links:
        link_domain = get_domain_name(link.get("href"))
        if link_domain != domain:
            hyper_links.remove(link)

    return hyper_links

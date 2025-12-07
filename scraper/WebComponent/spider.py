from typing import Literal
from bs4 import BeautifulSoup

from fetcher import RequestFactory
from utils import process_hyper_links


class Spider:
    def process_request(self, 
                        url: str, 
                        from_source: Literal["web", "api"], 
                        proxy: str = None):
        raise NotImplementedError
    
    def process_response(self, response: dict):
        raise NotImplementedError


class WebSpider(Spider):
    def __init__(self):
        self.bs4 = BeautifulSoup

    async def process_request(self, url: str, proxy: str = None) -> dict:
        request = RequestFactory("web")
        response = await request.get(url, proxy)
        return response

    def process_response(self, response: dict) -> dict:
        html_doc = response["page"].text
        soup = self.bs4(html_doc, "lxml")
        hyper_links = process_hyper_links(response["url"], soup.find_all("a"))
        data = {
            "html": soup,
            "hyper_links": hyper_links,
        }
        return data


class ApiSpider(Spider):
    pass


class SpiderFactory:
    def create(self, from_source: str):
        if from_source == "web":
            return WebSpider()
        elif from_source == "api":
            return ApiSpider()
        else:
            raise ValueError("Invalid source")
        
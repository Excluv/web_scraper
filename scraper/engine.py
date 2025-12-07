from typing import Self, Union, Optional

from WebComponent.spider import SpiderFactory
from WebComponent.fetcher import SingletonBrowser, Proxy
from ExtractorComponent.extractor import ContentExtractor
from dbinterface import DbInterface
from logger import ScraperLogger


class Node:
    def __init__(self, url: str):
        self.url = url
        self.depth = self._get_depth()

    def _get_depth(self) -> int:
        path = self.url.split("/")
        try:
            if "https" in path:
                depth = len(path[3:])
            else:
                depth = len(path[1:])
        except IndexError:
            depth = 0

        return depth
    
    def __lt__(self, node: Self) -> bool:
        return self.depth < node.depth
    
    def __gt__(self, node: Self) -> bool:
        return self.depth > node.depth


class HeapQueue:
    def __init__(self, entry: str):
        self.heapq = [Node(entry)]
        self.heap_size = 1

    def _swap(self, a: Node, b: Node) -> Union[Node, Node]:
        return b, a

    def _parent(self, idx: int) -> int:
        return int((idx - 1) / 2)

    def _left(self, idx: int) -> int:
        return int(2 * idx + 1)

    def _right(self, idx: int) -> int:
        return int(2 * idx + 2)

    def extract_min(self) -> Node:
        head = self.heapq[0]
        last = self.heap_size - 1
        self.heapq[0], self.heapq[last] = self._swap(self.heapq[0], self.heapq[last])
        self.min_heapify(0)
        return head
    
    def insert(self, url: str) -> None:
        self.heap_size += 1
        idx = self.heap_size - 1
        self.heapq[idx] = Node(url)
        while (idx != 0) and (self.heapq[self._parent(idx)] > self.heapq[idx]):
            self.heapq[idx], self.heapq[self._parent(idx)] = self._swap(self.heapq[idx], self.heapq[self._parent(idx)])
            idx = self._parent(idx)

    def min_heapify(self,idx: int) -> None:
        left = self._left(idx)
        right = self._right(idx)
        smallest = idx
        if left < self.heap_size and self.heapq[left] > self.heapq[idx]:
            smallest = left
        if right < self.heap_size and self.heapq[right] > self.heapq[idx]:
            smallest = right
        if smallest != idx:
            self.heapq[idx], self.heapq[smallest] = self._swap(self.heapq[idx], self.heapq[smallest])
            self.min_heapify(smallest)


class Sitemap:
    def __init__(self):
        self.heapq = HeapQueue()

    def push(self, url: str) -> None:
        self.heapq.insert(url)

    def pop(self) -> str:
        return self.heapq.extract_min().url
    
    def empty(self) -> bool:
        return bool(self.heapq.heap_size)


class SingletonSitemap:
    _instance = None

    def __new__(cls) -> Sitemap:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        
        return cls._instance

    def _initialize(self) -> None:
        self.sitemap = Sitemap()


class SpiderEngine:
    async def __init__(self, from_source: str, proxy: Proxy):
        self.proxy = proxy
        self.browser = await SingletonBrowser.get_instance(proxy.random())
        self.spider = SpiderFactory().create(from_source, proxy.random())
        self.site_map = SingletonSitemap()
        self.extractor = ContentExtractor()
        self.db = DbInterface()

    def expand_site_map(self, hyper_links: list):
        for link in hyper_links:
            self.site_map.insert(link)

    async def run(self, entry_url: str):
        self.site_map.push(entry_url)
        init_response = await self.spider.process_request(entry_url, self.proxy.random())
        init_response_data = self.spider.process_response(init_response)
        self.expand_site_map(init_response_data["hyper_links"])
        
        while not self.site_map.empty():
            url = self.site_map.pop()
            response = await self.spider.process_request(url, self.proxy.random())
            response_data = self.spider.process_response(response)
            self.expand_site_map(response["hyper_links"])

            content = self.extractor.process_content(response_data["html"])
            await self.db.save(content)

    async def close(self):
        await self.browser.close()

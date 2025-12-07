import random
from typing import Literal, Optional
from playwright.async_api import async_playwright
from fake_useragent import UserAgent
from datetime import datetime

from logger import ScraperLogger
from utils import get_domain_name


logger = ScraperLogger(__name__).logger

user_agent = UserAgent(browsers=["chrome"], os=["windows", "linux", "macos"])

class SingletonBrowser:
    _instance = None
    _initialized = False

    def __init__(self):
        if not SingletonBrowser._initialized:
            self.browser = None
            self.context = None
            SingletonBrowser._initialized = True

    @classmethod
    async def get_instance(cls, proxy: str = None):
        if cls._instance is None:
            cls._instance = SingletonBrowser()
            await cls._instance._initialize(proxy)
        
        return cls._instance
        
    async def _initialize(self, proxy: str = None):
        context_kwargs = {
            "viewport": {"width": 1920, "height": 1080},
            "user_agent": user_agent.random,
            "java_script_enabled": True,
            "bypass_csp": True,
            "ignore_https_errors": True,
            "locale": "en-US",
            "timezone_id": "Asia/Bangkok",
            "has_touch": False,
            "is_mobile": False,
        }
        if proxy:
            context_kwargs["proxy"] = {"server": proxy}
        
        playwright = async_playwright().start()
        self.browser = await playwright.chromium.launch(
            headless=True,
            args=[
                "--no-sanbox",
                "--disable-setupid-sanbox",
                "--disable-infobars",
                "--disable-blink-features=AutomationControlled",
                "--start-maximized",
            ]
        )
        self.context = await self.browser.new_context(**context_kwargs)

        # Stealth trick
        await self.context.add_innit_script("""
            Object.defineProperty(navigator, 'webdriver', {
                get: () => false,
            });
            window.chrome = { runtime: {}, app: {}, webstore: {} };
            Object.defineProperty(navigator, 'languages', { get: () => ['en-US', 'en'] });
            Object.defineProperty(navigator, 'plugins', {
                get: () => [1, 2, 3, 4, 5].map(i => ({ name: `plugin ${i}`, filename: `plugin${i}.dll`, description: '', mimeTypes: [] }))
            });
        """)
    
    async def new_page(self):
        return await self.context.new_page()
    
    async def close(self):
        if self.browser:
            await self.browser.close()

        self._instance = None
        SingletonBrowser._initialized = False


class Request:
    def prepare_context(self, proxy: str = None):
        raise NotImplementedError
    
    async def get(self, url: str, proxy: str = None):
        raise NotImplementedError
    

class WebRequest(Request):
    async def get(self, browser: SingletonBrowser, url: str, proxy: str = None) -> Optional[dict]:
        logger.info(f"GET requesting {url} via proxy {proxy}")
        page = await browser.new_page()
        try:
            # Bypass Cloudflare
            response = await page.goto(url, wait_until="networkidle", timeout=90000)
            if response.status == 200:
                data = {
                    "datetime": datetime.now().isoformat(),
                    "url": url,
                    "response": response,
                    "proxy": proxy or "direct",
                }
                logger.info("Page retrieved")
                return data
            else:
                logger.info(f"Failed to retrieve page, status code: {response.status} - reason: {response.status_text}")
                return None
        except Exception as err:
            logger.error("Failed to retrieve page, screenshot taken", exc_info=True)
            path = f"../logs/error_screenshots/{get_domain_name(url)}/error_{int(datetime.now().timestamp())}.png"
            await page.screenshot(path=path, full_page=True)
        finally:
            await page.close()


class ApiRequest(Request):
    pass


class RequestFactory:
    def create(self, from_source: Literal["web", "api"]):
        if from_source == "web":
            return WebRequest()
        elif from_source == "api":
            return ApiRequest()
        else:
            raise ValueError("Invalid source")


class Proxy:
    def __init__(self):
        self.proxy = []

    def random(self) -> Optional[str]:
        n_proxies = len(self.proxy)
        if n_proxies > 0:
            selected = random.randint(0, n_proxies - 1)
            return self.proxy[selected]
        else:
            return None

# scitex.browser.automation Architecture Proposal
**Date:** 2025-10-19
**Inspired by:** crawl4ai
**Goal:** Versatile browser automation with authentication handling

## Overview

Create a comprehensive browser automation system that:
1. **Handles authentication** (sessions, cookies, login flows)
2. **Extracts content** (markdown, JSON, structured data)
3. **Captures screenshots** (integrated with scitex.capture)
4. **Manages browser lifecycle** (context manager pattern)
5. **Supports strategies** (pluggable extraction, filtering, etc.)

## Architecture (crawl4ai-inspired)

```
scitex.browser/
├── automation/
│   ├── __init__.py
│   ├── AuthenticatedBrowser.py     # Main class (like AsyncWebCrawler)
│   ├── auth_strategy.py            # Authentication strategies
│   ├── session_manager.py          # Session/cookie management
│   ├── extraction_strategy.py      # Content extraction
│   ├── filter_strategy.py          # Content filtering
│   └── CookieHandler.py            # Existing cookie handler
├── core/
│   ├── browser_context.py          # Browser lifecycle management
│   └── page_manager.py             # Page operations
├── interaction/
│   └── (existing interaction utils)
├── debugging/
│   └── browser_logger              # (existing)
└── capture/
    └── (link to scitex.capture)
```

## Core Classes

### 1. AuthenticatedBrowser (Main Interface)

```python
from scitex.browser.automation import AuthenticatedBrowser
from scitex.browser.automation import (
    DjangoAuthStrategy,
    MarkdownExtractionStrategy,
)

# Usage pattern (like crawl4ai)
async with AuthenticatedBrowser(
    auth_strategy=DjangoAuthStrategy(
        login_url="http://127.0.0.1:8000/auth/login/",
        credentials={'username': 'user', 'password': 'pass'},
    ),
    extraction_strategy=MarkdownExtractionStrategy(),
    cache_sessions=True,
) as browser:
    # Automatically authenticated
    result = await browser.navigate("http://127.0.0.1:8000/new/")

    print(result.html)
    print(result.markdown)
    print(result.screenshot_path)
    print(result.extracted_data)
```

### 2. BrowserConfig

```python
@dataclass
class BrowserConfig:
    """Browser configuration (like crawl4ai's BrowserConfig)."""
    browser_type: str = "chromium"  # chromium, firefox, webkit
    headless: bool = True
    viewport: dict = field(default_factory=lambda: {"width": 1920, "height": 1080"})
    user_agent: Optional[str] = None
    proxy: Optional[ProxyConfig] = None
    locale: str = "en-US"
    timezone: str = "UTC"
    geolocation: Optional[dict] = None
    permissions: List[str] = field(default_factory=list)
    extra_args: List[str] = field(default_factory=list)
```

### 3. NavigationConfig

```python
@dataclass
class NavigationConfig:
    """Navigation configuration."""
    url: str
    wait_until: str = "networkidle"  # load, domcontentloaded, networkidle
    timeout: int = 30000
    wait_for_selector: Optional[str] = None
    scroll: bool = False
    screenshot: bool = True
    javascript: Optional[List[str]] = None  # Execute JS before extraction
```

### 4. AuthStrategy (Base Class)

```python
class AuthStrategy(ABC):
    """Base authentication strategy."""

    @abstractmethod
    async def authenticate(self, page: Page) -> bool:
        """Perform authentication. Returns True if successful."""
        pass

    @abstractmethod
    async def is_authenticated(self, page: Page) -> bool:
        """Check if currently authenticated."""
        pass

    @abstractmethod
    async def get_session_data(self) -> dict:
        """Get session data for persistence."""
        pass

    @abstractmethod
    async def restore_session(self, session_data: dict):
        """Restore from saved session data."""
        pass
```

### 5. Concrete Auth Strategies

```python
class DjangoAuthStrategy(AuthStrategy):
    """Django authentication strategy."""

    def __init__(
        self,
        login_url: str,
        credentials: dict,
        username_field: str = "#id_username",
        password_field: str = "#id_password",
        submit_button: str = "button[type='submit']",
        success_url_pattern: str = "**/core/**",
    ):
        self.login_url = login_url
        self.credentials = credentials
        self.username_field = username_field
        self.password_field = password_field
        self.submit_button = submit_button
        self.success_url_pattern = success_url_pattern

    async def authenticate(self, page: Page) -> bool:
        """Perform Django login."""
        await page.goto(self.login_url)

        # Fill credentials
        await fill_with_fallbacks_async(
            page,
            self.username_field,
            self.credentials['username']
        )
        await fill_with_fallbacks_async(
            page,
            self.password_field,
            self.credentials['password']
        )

        # Submit
        await click_with_fallbacks_async(page, self.submit_button, "Log In")

        # Wait for redirect
        try:
            await page.wait_for_url(
                self.success_url_pattern,
                timeout=5000
            )
            return True
        except TimeoutError:
            return False

    async def is_authenticated(self, page: Page) -> bool:
        """Check if on login page."""
        return "login" not in page.url.lower()

    async def get_session_data(self) -> dict:
        """Get Django session cookies."""
        context = page.context
        cookies = await context.cookies()
        return {
            'cookies': cookies,
            'storage_state': await context.storage_state(),
        }

    async def restore_session(self, session_data: dict):
        """Restore Django session."""
        await context.add_cookies(session_data['cookies'])


class OAuth2Strategy(AuthStrategy):
    """OAuth2 authentication strategy."""
    # Implementation for OAuth flows
    pass


class APIKeyStrategy(AuthStrategy):
    """API key authentication strategy."""
    # Implementation for API key auth
    pass


class CertificateStrategy(AuthStrategy):
    """Client certificate authentication."""
    # Implementation for cert-based auth
    pass
```

### 6. ExtractionStrategy

```python
class ExtractionStrategy(ABC):
    """Base extraction strategy (like crawl4ai)."""

    @abstractmethod
    async def extract(self, page: Page) -> dict:
        """Extract data from page."""
        pass


class MarkdownExtractionStrategy(ExtractionStrategy):
    """Extract content as markdown."""

    async def extract(self, page: Page) -> dict:
        html = await page.content()
        # Use crawl4ai's html2text or similar
        markdown = html_to_markdown(html)
        return {'markdown': markdown}


class JSONExtractionStrategy(ExtractionStrategy):
    """Extract JSON data via CSS/XPath selectors."""

    def __init__(self, schema: dict):
        self.schema = schema

    async def extract(self, page: Page) -> dict:
        """
        schema = {
            'title': 'h1.title',
            'description': '.description',
            'links': {
                'selector': 'a.link',
                'multiple': True,
                'fields': {
                    'text': 'text',
                    'url': 'href',
                }
            }
        }
        """
        result = {}
        for key, selector in self.schema.items():
            if isinstance(selector, dict):
                # Nested extraction
                elements = await page.query_selector_all(selector['selector'])
                items = []
                for el in elements:
                    item = {}
                    for field, attr in selector['fields'].items():
                        if attr == 'text':
                            item[field] = await el.text_content()
                        else:
                            item[field] = await el.get_attribute(attr)
                    items.append(item)
                result[key] = items
            else:
                # Simple extraction
                element = await page.query_selector(selector)
                result[key] = await element.text_content() if element else None
        return result


class LLMExtractionStrategy(ExtractionStrategy):
    """LLM-based extraction."""

    def __init__(self, prompt: str, model: str = "gpt-4"):
        self.prompt = prompt
        self.model = model

    async def extract(self, page: Page) -> dict:
        html = await page.content()
        # Use LLM to extract structured data
        response = await llm_extract(html, self.prompt, self.model)
        return response
```

### 7. SessionManager

```python
class SessionManager:
    """Manage browser sessions and cookies."""

    def __init__(
        self,
        session_dir: str = "~/.scitex/browser/sessions",
        encryption_key: Optional[str] = None,
    ):
        self.session_dir = Path(os.path.expanduser(session_dir))
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.encryption_key = encryption_key

    async def save_session(
        self,
        session_id: str,
        cookies: List[dict],
        storage_state: dict,
    ):
        """Save session to disk."""
        session_file = self.session_dir / f"{session_id}.json"
        data = {
            'cookies': cookies,
            'storage_state': storage_state,
            'timestamp': time.time(),
        }

        if self.encryption_key:
            data = encrypt(data, self.encryption_key)

        with open(session_file, 'w') as f:
            json.dump(data, f)

    async def load_session(self, session_id: str) -> Optional[dict]:
        """Load session from disk."""
        session_file = self.session_dir / f"{session_id}.json"
        if not session_file.exists():
            return None

        with open(session_file, 'r') as f:
            data = json.load(f)

        if self.encryption_key:
            data = decrypt(data, self.encryption_key)

        return data

    async def delete_session(self, session_id: str):
        """Delete saved session."""
        session_file = self.session_dir / f"{session_id}.json"
        if session_file.exists():
            session_file.unlink()

    async def list_sessions(self) -> List[str]:
        """List all saved sessions."""
        return [f.stem for f in self.session_dir.glob("*.json")]
```

### 8. BrowserResult (like CrawlResult)

```python
@dataclass
class BrowserResult:
    """Result from browser navigation (like crawl4ai's CrawlResult)."""

    url: str
    html: str
    markdown: Optional[str] = None
    extracted_data: Optional[dict] = None
    screenshot_path: Optional[str] = None
    success: bool = True
    error: Optional[str] = None
    metadata: dict = field(default_factory=dict)

    # Session info
    cookies: List[dict] = field(default_factory=list)
    authenticated: bool = False

    # Performance
    load_time: float = 0.0
    response_status: int = 200

    # Links
    links: List[str] = field(default_factory=list)
    internal_links: List[str] = field(default_factory=list)
    external_links: List[str] = field(default_factory=list)
```

## Main Class Implementation

```python
class AuthenticatedBrowser:
    """
    Versatile browser automation with authentication.

    Usage:
        # Simple usage
        async with AuthenticatedBrowser() as browser:
            result = await browser.navigate("https://example.com")

        # With authentication
        async with AuthenticatedBrowser(
            auth_strategy=DjangoAuthStrategy(...),
        ) as browser:
            result = await browser.navigate("https://example.com/protected/")

        # Long-running
        browser = AuthenticatedBrowser()
        await browser.start()

        result1 = await browser.navigate("url1")
        result2 = await browser.navigate("url2")

        await browser.close()
    """

    def __init__(
        self,
        browser_config: Optional[BrowserConfig] = None,
        auth_strategy: Optional[AuthStrategy] = None,
        extraction_strategy: Optional[ExtractionStrategy] = None,
        session_manager: Optional[SessionManager] = None,
        cache_sessions: bool = True,
        auto_screenshot: bool = True,
        logger: Optional[AsyncLogger] = None,
    ):
        self.browser_config = browser_config or BrowserConfig()
        self.auth_strategy = auth_strategy
        self.extraction_strategy = extraction_strategy
        self.session_manager = session_manager or SessionManager()
        self.cache_sessions = cache_sessions
        self.auto_screenshot = auto_screenshot
        self.logger = logger or AsyncLogger()

        self.browser = None
        self.context = None
        self.page = None
        self.authenticated = False

    async def __aenter__(self):
        """Context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, *args):
        """Context manager exit."""
        await self.close()

    async def start(self):
        """Start browser explicitly."""
        async with async_playwright() as p:
            # Launch browser
            self.playwright = p
            browser_type = getattr(p, self.browser_config.browser_type)
            self.browser = await browser_type.launch(
                headless=self.browser_config.headless,
                args=self.browser_config.extra_args,
            )

            # Create context
            self.context = await self.browser.new_context(
                viewport=self.browser_config.viewport,
                user_agent=self.browser_config.user_agent,
                locale=self.browser_config.locale,
                timezone_id=self.browser_config.timezone,
            )

            # Load saved session if available
            if self.cache_sessions and self.auth_strategy:
                session_id = self._get_session_id()
                session_data = await self.session_manager.load_session(session_id)
                if session_data:
                    await self.context.add_cookies(session_data['cookies'])
                    self.authenticated = True

            # Create page
            self.page = await self.context.new_page()

            await self.logger.info("Browser started")

    async def close(self):
        """Close browser explicitly."""
        if self.page:
            await self.page.close()
        if self.context:
            await self.context.close()
        if self.browser:
            await self.browser.close()

        await self.logger.info("Browser closed")

    async def navigate(
        self,
        url: str,
        config: Optional[NavigationConfig] = None,
    ) -> BrowserResult:
        """
        Navigate to URL and extract content.

        Handles authentication automatically.
        """
        config = config or NavigationConfig(url=url)

        try:
            # Check if authentication needed
            if self.auth_strategy and not self.authenticated:
                await self._authenticate()

            # Navigate
            await self.page.goto(
                url,
                wait_until=config.wait_until,
                timeout=config.timeout,
            )

            # Check if redirected to login (session expired)
            if self.auth_strategy and not await self.auth_strategy.is_authenticated(self.page):
                await self._authenticate()
                await self.page.goto(url, wait_until=config.wait_until)

            # Wait for selector if specified
            if config.wait_for_selector:
                await self.page.wait_for_selector(
                    config.wait_for_selector,
                    timeout=config.timeout,
                )

            # Execute JavaScript if specified
            if config.javascript:
                for script in config.javascript:
                    await self.page.evaluate(script)

            # Scroll if requested
            if config.scroll:
                await self._auto_scroll()

            # Extract content
            html = await self.page.content()

            # Extract markdown if strategy provided
            markdown = None
            extracted_data = None
            if self.extraction_strategy:
                extraction_result = await self.extraction_strategy.extract(self.page)
                if isinstance(self.extraction_strategy, MarkdownExtractionStrategy):
                    markdown = extraction_result.get('markdown')
                else:
                    extracted_data = extraction_result

            # Take screenshot if enabled
            screenshot_path = None
            if self.auto_screenshot or config.screenshot:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                url_slug = url.replace("://", "_").replace("/", "_")
                screenshot_path = f"$SCITEX_DIR/capture/{timestamp}-{url_slug}.jpg"
                await self.page.screenshot(
                    path=os.path.expanduser(screenshot_path),
                    quality=90,
                )

            # Get links
            links = await self._extract_links()

            # Build result
            result = BrowserResult(
                url=url,
                html=html,
                markdown=markdown,
                extracted_data=extracted_data,
                screenshot_path=screenshot_path,
                success=True,
                cookies=await self.context.cookies(),
                authenticated=self.authenticated,
                links=links,
            )

            # Save session if caching enabled
            if self.cache_sessions and self.authenticated:
                await self._save_session()

            return result

        except Exception as e:
            return BrowserResult(
                url=url,
                html="",
                success=False,
                error=str(e),
            )

    async def _authenticate(self):
        """Perform authentication."""
        success = await self.auth_strategy.authenticate(self.page)
        if success:
            self.authenticated = True
            await self.logger.info("Authentication successful")
        else:
            self.authenticated = False
            await self.logger.error("Authentication failed")
            raise AuthenticationError("Failed to authenticate")

    async def _save_session(self):
        """Save current session."""
        session_id = self._get_session_id()
        session_data = await self.auth_strategy.get_session_data()
        await self.session_manager.save_session(
            session_id,
            session_data['cookies'],
            session_data['storage_state'],
        )

    def _get_session_id(self) -> str:
        """Generate session ID based on auth strategy."""
        # Use URL domain + auth type as session ID
        return f"{self.auth_strategy.__class__.__name__}"

    async def _auto_scroll(self):
        """Auto-scroll page to load dynamic content."""
        await self.page.evaluate("""
            async () => {
                await new Promise((resolve) => {
                    let totalHeight = 0;
                    const distance = 100;
                    const timer = setInterval(() => {
                        window.scrollBy(0, distance);
                        totalHeight += distance;
                        if (totalHeight >= document.body.scrollHeight) {
                            clearInterval(timer);
                            resolve();
                        }
                    }, 100);
                });
            }
        """)

    async def _extract_links(self) -> List[str]:
        """Extract all links from page."""
        links = await self.page.evaluate("""
            () => Array.from(document.querySelectorAll('a')).map(a => a.href)
        """)
        return links
```

## Usage Examples

### Example 1: Simple Authenticated Navigation

```python
from scitex.browser.automation import AuthenticatedBrowser, DjangoAuthStrategy

async def main():
    async with AuthenticatedBrowser(
        auth_strategy=DjangoAuthStrategy(
            login_url="http://127.0.0.1:8000/auth/login/",
            credentials={
                'username': os.getenv('SCITEX_CLOUD_USERNAME'),
                'password': os.getenv('SCITEX_CLOUD_PASSWORD'),
            },
        ),
    ) as browser:
        # Navigate to protected page
        result = await browser.navigate("http://127.0.0.1:8000/new/")

        print(f"Success: {result.success}")
        print(f"Screenshot: {result.screenshot_path}")
        print(f"Authenticated: {result.authenticated}")
```

### Example 2: Data Extraction

```python
from scitex.browser.automation import (
    AuthenticatedBrowser,
    DjangoAuthStrategy,
    JSONExtractionStrategy,
)

schema = {
    'title': 'h1',
    'projects': {
        'selector': '.project-card',
        'multiple': True,
        'fields': {
            'name': '.project-name',
            'description': '.project-description',
        }
    }
}

async with AuthenticatedBrowser(
    auth_strategy=DjangoAuthStrategy(...),
    extraction_strategy=JSONExtractionStrategy(schema),
) as browser:
    result = await browser.navigate("http://127.0.0.1:8000/projects/")

    print(result.extracted_data)
    # {
    #     'title': 'My Projects',
    #     'projects': [
    #         {'name': 'Project 1', 'description': '...'},
    #         {'name': 'Project 2', 'description': '...'},
    #     ]
    # }
```

### Example 3: Multiple Pages (Crawling)

```python
async with AuthenticatedBrowser(
    auth_strategy=DjangoAuthStrategy(...),
) as browser:
    # First page
    result1 = await browser.navigate("http://127.0.0.1:8000/projects/")

    # Extract links
    for link in result1.links:
        if '/projects/' in link:
            result = await browser.navigate(link)
            print(f"Crawled: {result.url}")
```

## Integration with scitex.capture

Update scitex.capture to use AuthenticatedBrowser:

```python
# In scitex/capture/utils.py

async def snap_authenticated(
    url: str,
    username: str = None,
    password: str = None,
    **kwargs
):
    """
    Capture screenshot with authentication.

    Uses scitex.browser.automation under the hood.
    """
    from scitex.browser.automation import AuthenticatedBrowser, DjangoAuthStrategy

    username = username or os.getenv('SCITEX_CLOUD_USERNAME')
    password = password or os.getenv('SCITEX_CLOUD_PASSWORD')

    async with AuthenticatedBrowser(
        auth_strategy=DjangoAuthStrategy(
            login_url="http://127.0.0.1:8000/auth/login/",
            credentials={'username': username, 'password': password},
        ),
        auto_screenshot=True,
    ) as browser:
        result = await browser.navigate(url)
        return result.screenshot_path
```

## MCP Integration

```python
# In scitex/capture/mcp_server.py

async def capture_screenshot_authenticated(
    self,
    url: str,
    username: str = None,
    password: str = None,
    **kwargs
):
    """MCP tool for authenticated screenshot capture."""
    path = await snap_authenticated(url, username, password)
    return {
        "success": True,
        "path": path,
        "category": "authenticated",
    }
```

## Benefits

1. **Versatile:** Supports multiple auth types (Django, OAuth2, API keys, certs)
2. **Session Management:** Automatic cookie persistence
3. **Content Extraction:** Multiple strategies (markdown, JSON, LLM)
4. **Screenshot Integration:** Built-in scitex.capture integration
5. **AI-Friendly:** Perfect for AI agents to interact with web apps
6. **Crawl4ai-Compatible:** Similar API patterns for easy learning

## Implementation Plan

1. ✅ Design architecture (this document)
2. ⏳ Implement core classes
3. ⏳ Add auth strategies
4. ⏳ Add extraction strategies
5. ⏳ Integrate with scitex.capture
6. ⏳ Add MCP tools
7. ⏳ Write tests
8. ⏳ Create examples

---

**Status:** Architecture complete, ready for implementation
**Estimated effort:** 1-2 days
**Priority:** High - enables full AI agent automation

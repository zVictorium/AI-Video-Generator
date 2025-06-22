"""
Google Authentication Module for Browser Automation
Provides functions to authenticate with Google accounts and handle OAuth flows.
"""

import os
import random
import time
from playwright.sync_api import sync_playwright

# Configuration constants
HEADLESS_MODE = True
DEBUG_SCREENSHOTS = False


def human_like_delay(min_sec=0.5, max_sec=2.0):
    """Add random delay to simulate human behavior"""
    time.sleep(random.uniform(min_sec, max_sec))


# === Browser Creation and Configuration ===


def get_browser_args():
    """Return optimized Chromium arguments to avoid detection"""
    return [
        "--disable-blink-features=AutomationControlled",
        "--disable-features=IsolateOrigins,site-per-process",
        "--disable-dev-shm-usage",
        "--disable-popup-blocking",
        "--disable-extensions",
        "--disable-background-timer-throttling",
        "--disable-backgrounding-occluded-windows",
        "--disable-renderer-backgrounding",
        "--metrics-recording-only",
        "--no-sandbox",
        "--window-size=1920,1080",
    ]


def get_stealth_script():
    """Return JavaScript to help avoid bot detection"""
    return """
        // Mask automation indicators
        Object.defineProperty(navigator, 'webdriver', {
            get: () => false
        });
        
        // Add language preferences 
        Object.defineProperty(navigator, 'languages', {
            get: () => ['en-US', 'en', 'es']
        });
        
        // Add fake plugins
        if (navigator.plugins) {
            Object.defineProperty(navigator, 'plugins', {
                get: () => {
                    return [
                        { name: 'PDF Viewer', description: 'Portable Document Format', filename: 'internal-pdf-viewer' },
                        { name: 'Chrome PDF Viewer', description: 'Portable Document Format', filename: 'internal-pdf-viewer' },
                        { name: 'WebKit built-in PDF', description: 'Portable Document Format', filename: 'internal-pdf-viewer' }
                    ];
                }
            });
        }
        
        // Hide automation artifacts
        delete window.cdc_adoQpoasnfa76pfcZLmcfl_;
        
        // Add expected browser objects
        if (!window.chrome) {
            window.chrome = { runtime: {} };
        }
    """


def create_stealth_browser(playwright, user_data_dir=None, headless=False):
    """
    Create a browser with anti-detection measures

    Args:
        playwright: Playwright instance
        user_data_dir: Directory for persistent browser data
        headless: Whether to run in headless mode

    Returns:
        tuple: (browser, context, page)
    """
    # Setup user data directory
    if not user_data_dir:
        user_data_dir = os.path.join(os.path.expanduser("~"), ".chromium_user_data")

    if not os.path.exists(user_data_dir):
        os.makedirs(user_data_dir)

    # Select random user agent
    user_agents = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36",
    ]
    user_agent = random.choice(user_agents)

    # Launch browser with stealth settings
    context = playwright.chromium.launch_persistent_context(
        user_data_dir=user_data_dir,
        headless=headless,
        viewport={"width": 1920, "height": 1080},
        locale="en-US",
        timezone_id="America/New_York",
        permissions=["geolocation", "notifications"],
        args=get_browser_args(),
        user_agent=user_agent,
    )

    browser = context.browser
    page = context.new_page()
    page.add_init_script(get_stealth_script())

    # Initialize browser history
    try:
        initial_site = random.choice(
            [
                "https://www.wikipedia.org/",
                "https://www.reddit.com/",
                "https://www.cnn.com/",
            ]
        )
        page.goto(initial_site, timeout=10000)
        human_like_delay()
        page.mouse.wheel(delta_x=0, delta_y=2000)
        human_like_delay()
        page.mouse.wheel(delta_x=0, delta_y=-1000)
    except Exception:
        pass

    return browser, context, page


# === Login Detection and Authentication ===


def detect_login_status_from_page(page):
    """
    Check if user is logged into Google based on page content

    Returns:
        bool: Login status
    """
    try:
        if (
            page.query_selector('a[href*="myaccount.google.com"]')
            or page.query_selector("img[data-email]")
            or page.query_selector('img[alt="Google Account"]')
            or page.query_selector('a[href*="accounts.google.com/SignOut"]')
        ):
            return True
        return False
    except Exception:
        return False


def is_google_logged_in(page, navigate=False):
    """
    Check if the browser is logged in with Google

    Args:
        page: Playwright page object
        navigate: If True, navigates to check login status

    Returns:
        bool: Login status
    """
    try:
        # First try detecting from current page if on Google domain
        if "google.com" in page.url and detect_login_status_from_page(page):
            return True

        # Navigate to Google if requested or perform lightweight check
        page.goto("https://www.google.com/", wait_until="networkidle")
        return detect_login_status_from_page(page)
    except Exception:
        return False


def login_to_google(page, context, playwright, timeout=300):
    """
    Let the user log in to Google manually

    Args:
        page: Playwright page object
        context: Playwright browser context
        playwright: Playwright instance
        timeout: Maximum wait time in seconds

    Returns:
        tuple: (bool success, page object, context object)
    """
    global HEADLESS_MODE

    # Check if already logged in
    try:
        if detect_login_status_from_page(page):
            print("Already logged in to Google")
            return True, page, context
    except Exception:
        pass

    # Switch to visible browser for login if needed
    if HEADLESS_MODE:
        print("\nSwitching to visible browser for authentication...\n")
        try:
            context.close()
        except Exception:
            pass

        user_data_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "chromium"
        )
        _, context, page = create_stealth_browser(
            playwright, user_data_dir=user_data_dir, headless=False
        )

    # Navigate to sign-in page
    try:
        page.goto("https://accounts.google.com/signin", wait_until="domcontentloaded")
        human_like_delay(1.0, 2.0)
    except Exception:
        pass

    print("\n" + "=" * 50)
    print("Please log in to your Google account in the browser")
    print("The program will wait until login completes")
    print("=" * 50 + "\n")

    # Wait for login completion
    start_time = time.time()
    login_successful = False

    while time.time() - start_time < timeout:
        try:
            current_url = page.url

            try:
                page.wait_for_load_state("networkidle", timeout=5000)
            except:
                pass

            # Check for login success indicators
            if "myaccount.google.com" in current_url:
                login_successful = True
                break

            try:
                if detect_login_status_from_page(page):
                    login_successful = True
                    break
            except Exception:
                pass

            time.sleep(2)

        except Exception:
            time.sleep(2)

    if not login_successful:
        print("Login timeout reached. Please try again.")

    if HEADLESS_MODE and login_successful:
        print("\nLogin successful. Keeping visible browser for this session.")

    return login_successful, page, context


# === YouTube OAuth Handling ===


def open_youtube_oauth(page, oauth_url, account_index=0):
    """
    Open YouTube OAuth URL and select a Google account profile

    Args:
        page: Playwright page object
        oauth_url: OAuth URL to navigate to
        account_index: Index of the account to select (0 for first account)

    Returns:
        bool: True if successful
    """
    try:
        # Navigate to the OAuth URL
        print("Opening YouTube OAuth URL...")
        page.goto(oauth_url, wait_until="networkidle")
        human_like_delay()
        debug_screenshot(page, "oauth_page")

        # Find profiles using structured DOM analysis
        profiles, action_buttons = find_profile_elements(page)

        # Print available profiles
        print("Found profiles:")
        for i, profile_info in enumerate(profiles):
            print(f"  [{i}] {profile_info['name']}")

        # Select profile or use "add account" action
        if profiles and len(profiles) > account_index:
            profile = profiles[account_index]
            print(f"Selecting profile: {profile['name']}")
            page.click(profile["selector"])
            human_like_delay(1.0, 3.0)

            handle_verification_warning(page)
            handle_permissions_page(page)
            return True

        elif account_index >= len(profiles) and account_index > 0 and action_buttons:
            # Try to use "Another account" option
            print(
                f"Profile index {account_index} not available. Clicking '{action_buttons[0]['name']}' button."
            )
            page.click(action_buttons[0]["selector"])
            human_like_delay(2.0, 4.0)
            return True
        else:
            # Fallback to JavaScript selection
            try_select_profile_with_js(page, account_index)

        # If we reached this point but didn't return yet, we couldn't find or click a profile
        print(
            "Could not select a profile automatically. Manual intervention may be required."
        )
        debug_screenshot(page, "oauth_failed")
        return False

    except Exception as e:
        print(f"Error during YouTube OAuth: {e}")
        debug_screenshot(page, "oauth_error")
        return False


def try_select_profile_with_js(page, account_index):
    """Use JavaScript to select a profile by index"""
    try:
        success = page.evaluate(
            f"""
            () => {{
                const profileElement = document.querySelector('.VV3oRb[data-item-index="{account_index}"]');
                if (profileElement) {{
                    profileElement.click();
                    return true;
                }}
                return false;
            }}
        """
        )

        if success:
            print(f"Clicked profile at index {account_index} using JavaScript")
            human_like_delay(1.0, 3.0)
            handle_verification_warning(page)
            return True

        # Try to find "Use another account" button for account_index > 0
        if account_index > 0:
            use_another_selectors = [
                "div.AsY17b:has-text('Use another account')",
                "div.AsY17b:has-text('Usar otra cuenta')",
                ".B682ne .VV3oRb",
                "[jsname='rwl3qc']",
            ]

            for selector in use_another_selectors:
                button = page.query_selector(selector)
                if button:
                    button.click()
                    human_like_delay(2.0, 4.0)
                    return True
    except Exception:
        pass

    return False


def find_profile_elements(page):
    """Find profile elements and action buttons on login page"""
    profiles = []
    action_buttons = []

    try:
        # Look for profile elements with data-item-index attribute
        elements = page.query_selector_all(".VV3oRb[data-item-index]")

        for element in elements:
            try:
                index = element.get_attribute("data-item-index")

                # Get name and email/service info
                name_element = element.query_selector(".pGzURd")
                name = name_element.inner_text() if name_element else "Unknown profile"

                email_element = element.query_selector(".yAlK0b")
                service_element = element.query_selector(".H2oig")

                if email_element:
                    identifier = email_element.inner_text()
                elif service_element:
                    identifier = f"{name} ({service_element.inner_text()})"
                else:
                    identifier = name

                # Create selector for clicking
                selector = f".VV3oRb[data-item-index='{index}']"

                profiles.append(
                    {
                        "index": index,
                        "name": name,
                        "identifier": identifier,
                        "selector": selector,
                    }
                )
            except Exception:
                pass

        # Find "Use another account" option
        use_another = page.query_selector(".aZvCDf.B682ne .VV3oRb, [jsname='rwl3qc']")
        if use_another:
            try:
                text_element = use_another.query_selector(".AsY17b")
                text = (
                    text_element.inner_text() if text_element else "Use another account"
                )

                action_buttons.append(
                    {
                        "type": "action",
                        "name": text,
                        "selector": ".aZvCDf.B682ne .VV3oRb, [jsname='rwl3qc']",
                    }
                )
            except Exception:
                pass

    except Exception:
        pass

    return profiles, action_buttons


def handle_verification_warning(page):
    """
    Handle the Google verification warning page

    Args:
        page: Playwright page object

    Returns:
        bool: True if handled successfully
    """
    try:
        # Allow page to stabilize
        try:
            page.wait_for_load_state("networkidle", timeout=8000)
        except Exception:
            pass

        human_like_delay(2.0, 3.5)
        debug_screenshot(page, "verification_page")

        # Detect warning page
        warning_detected = check_for_warning_page(page)

        # Handle warning page if detected
        if warning_detected:
            print("Detected verification warning page")

            # Try click with JavaScript first
            result = page.evaluate(
                """
                () => {
                    const buttons = Array.from(document.querySelectorAll('button'));
                    for (const button of buttons) {
                        if (button.innerText.includes('Continuar') || button.innerText.includes('Continue')) {
                            button.click();
                            return true;
                        }
                    }
                    return false;
                }
            """
            )

            if result:
                print("Clicked continue button using JavaScript")
                human_like_delay(1.5, 3.0)
                return True

            # Try specific selectors
            button_selectors = [
                "button.VfPpkd-LgbsSe:has(span:text('Continuar'))",
                "button.VfPpkd-LgbsSe:has(span.VfPpkd-vQzf8d:text('Continuar'))",
                "button.VfPpkd-LgbsSe.ksBjEc",
                ".VfPpkd-dgl2Hf-ppHlrf-sM5MNb button",
                "button:has-text('Continuar')",
                "button:has-text('Continue')",
                "[jsname='eBSUOb'] button",
            ]

            for selector in button_selectors:
                try:
                    if page.query_selector(selector):
                        page.click(selector)
                        human_like_delay(1.5, 3.0)
                        try:
                            page.wait_for_load_state("networkidle", timeout=5000)
                        except:
                            pass
                        return True
                except Exception:
                    pass
        else:
            print("No verification warning page detected, continuing...")

        return True

    except Exception as e:
        print(f"Error handling verification warning: {e}")
        return True


def check_for_warning_page(page):
    """Check if current page is a verification warning page"""
    try:
        # Check page content for warning text
        page_content = page.content()
        warning_texts = [
            "Google no ha verificado esta aplicación",
            "Google hasn't verified this app",
        ]

        for text in warning_texts:
            if text in page_content:
                return True

        # Check page title
        title = page.title().lower()
        if "verificado" in title or "verified" in title:
            return True

        # Check for warning elements with JavaScript
        js_result = page.evaluate(
            """
            () => {
                const headings = Array.from(document.querySelectorAll('h1, h2, .OyEIQ, .kVr7Lc'));
                for (const heading of headings) {
                    const text = heading.innerText.toLowerCase();
                    if (text.includes('verified') || text.includes('verificado')) {
                        return true;
                    }
                }
                return false;
            }
        """
        )

        return js_result

    except Exception:
        return False


def handle_permissions_page(page):
    """Handle OAuth permissions selection page"""
    try:
        # Wait for the page to stabilize
        try:
            page.wait_for_load_state("networkidle", timeout=5000)
        except Exception:
            pass

        human_like_delay(1.5, 2.5)
        debug_screenshot(page, "permissions_page")

        # Check if we're on the permissions page
        is_permissions_page = detect_permissions_page(page)

        if is_permissions_page:
            print("Detected OAuth permissions selection page")

            # Try to click "Select everything" checkbox
            checkbox_selectors = [
                'input[aria-label="Seleccionar todo"]',
                "#i1",
                'div[jsname="FkQz1b"]',
                '.VfPpkd-MPu53c:has(+ label:text("Seleccionar todo"))',
            ]

            for selector in checkbox_selectors:
                try:
                    if page.query_selector(selector):
                        page.click(selector)
                        print(f"Clicked checkbox with selector: {selector}")
                        break
                except Exception:
                    pass

            human_like_delay(1.0, 2.0)

            # Click the Continue button
            continue_selectors = [
                "#submit_approve_access button",
                'button:has(span:text("Continuar"))',
                '[jsname="uRHG6"] button',
                'button.VfPpkd-LgbsSe span:text("Continuar")',
            ]

            for selector in continue_selectors:
                try:
                    if page.query_selector(selector):
                        page.click(selector)
                        human_like_delay(1.5, 3.0)
                        return True
                except Exception:
                    pass

            # Use JavaScript as a fallback
            page.evaluate(
                """
                () => {
                    const buttons = Array.from(document.querySelectorAll('button'));
                    const continueButton = buttons.find(btn => 
                        btn.innerText.includes('Continuar') || 
                        btn.innerText.includes('Continue')
                    );
                    
                    if (continueButton) {
                        continueButton.click();
                        return true;
                    }
                    
                    const approveButton = document.querySelector('#submit_approve_access button');
                    if (approveButton) {
                        approveButton.click();
                        return true;
                    }
                }
            """
            )

            human_like_delay(1.5, 3.0)
        else:
            print("No permissions page detected, continuing...")

        return True

    except Exception:
        return True


def detect_permissions_page(page):
    """Detect if current page is permissions selection page"""
    try:
        # Check for the "Select everything" checkbox
        checkbox = page.query_selector('input[aria-label="Seleccionar todo"], #i1')
        if checkbox:
            return True

        # Check for heading text
        js_result = page.evaluate(
            """
            () => {
                const headings = Array.from(document.querySelectorAll('h1, h2'));
                for (const heading of headings) {
                    const text = heading.innerText.toLowerCase();
                    if (text.includes('acceder') || text.includes('selecciona') || 
                        text.includes('access') || text.includes('select')) {
                        return true;
                    }
                }
                return false;
            }
        """
        )

        return js_result
    except Exception:
        return False


# === Utility Functions ===


def debug_screenshot(page, name_prefix):
    """Take a debugging screenshot if debug mode is enabled"""
    if not DEBUG_SCREENSHOTS:
        return

    try:
        debug_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "chromium/screenshots"
        )
        if not os.path.exists(debug_dir):
            os.makedirs(debug_dir)

        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filepath = os.path.join(debug_dir, f"{name_prefix}_{timestamp}.png")

        page.screenshot(path=filepath)
        print(f"Debug screenshot saved to: {filepath}")
    except Exception:
        pass


def accept_oath_url(oauth_url):
    """Run Google login check and authentication process"""
    with sync_playwright() as p:
        print("Launching browser...")
        user_data_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "chromium"
        )
        _, context, page = create_stealth_browser(
            p, user_data_dir=user_data_dir, headless=HEADLESS_MODE
        )

        try:
            # Check login status and authenticate if needed
            if not is_google_logged_in(page):
                print("Not logged in, starting authentication...")
                login_success, page, context = login_to_google(page, context, p)
                if not login_success:
                    print("Login failed. Exiting.")
                    return
            else:
                print("Already logged in to Google")

            open_youtube_oauth(page, oauth_url, 1)

            print("Closing browser in 2 seconds...")
            time.sleep(2)
        finally:
            context.close()


if __name__ == "__main__":
    accept_oath_url("https://accounts.google.com/o/oauth2/auth?response_type=code&client_id=301135955017-15qn4blp21jk8ullq0asd3tq35im6dab.apps.googleusercontent.com&redirect_uri=http%3A%2F%2Flocalhost%3A53407%2F&scope=https%3A%2F%2Fwww.googleapis.com%2Fauth%2Fyoutube.upload+https%3A%2F%2Fwww.googleapis.com%2Fauth%2Fyoutube.force-ssl&state=mB5mo4nRqjV7qLiCCCP0xVrMFBDwn5&prompt=consent&access_type=offline")

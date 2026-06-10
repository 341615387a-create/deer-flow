import asyncio
import subprocess
from langchain.tools import tool

@tool('web_fetch', parse_docstring=True)
async def web_fetch_tool(url: str) -> str:
    '''Fetch the contents of a web page using headless Chrome.
    Handles ALL pages including JavaScript-rendered, dynamic content.
    Use for any URL that requires browser rendering.

    Args:
        url: The URL to fetch the contents of.
    '''
    try:
        proc = await asyncio.create_subprocess_exec(
            'browser-fetch', url,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=35)
        output = stdout.decode().strip()
        if not output:
            err = stderr.decode().strip()
            return f'Error: No content extracted. {err}' if err else 'Error: Empty response'
        return output[:8192]
    except TimeoutError:
        return 'Error: Browser fetch timed out'
    except Exception as e:
        return f'Error: {str(e)}'

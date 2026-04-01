from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import HTMLResponse

from ..services.oauth2 import OAuthClient


router = APIRouter()

# Temporary store for OAuth state and verifiers
# In production, use Redis or a proper session store
oauth_state_store = {}

@router.get("/oauth/callback")
async def get(request: Request):
    """OAuth callback that returns HTML to communicate token back to parent window"""
    # 1. Get parameters from the URL
    code = request.query_params.get("code")
    state = request.query_params.get("state")

    # 2. Verify state to prevent CSRF attacks and retrieve stored data
    oauth_data = oauth_state_store.pop(state, None)
    if not oauth_data:
        return HTMLResponse(content="""
            <!DOCTYPE html>
            <html>
            <head><title>Authentication Error</title></head>
            <body>
                <h2>Authentication Error</h2>
                <p>Invalid state or session expired. Please close this window and try again.</p>
            </body>
            </html>
        """, status_code=400)

    verifier = oauth_data["verifier"]
    oauth_client = oauth_data["oauth_client"]

    # 3. Exchange the code for the actual Access Token
    token_endpoint = "https://raul-cabello.ngrok.app/oidc/token"
    redirect_uri = "http://localhost:8000/callback"

    try:
        token = await oauth_client.fetch_token(
            token_endpoint=token_endpoint,
            authorization_response=str(request.url),
            redirect_uri=redirect_uri,
            verifier=verifier
        )

        access_token = token["access_token"]
        # setTimeout(() => window.close(), 500);
        # Return HTML that sends token to parent and closes popup
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head><title>Authentication Successful</title></head>
        <body>
            <h2>Authentication Successful!</h2>
            <p>Closing window...</p>
            <script>
                // Send the access token to the parent window
                if (window.opener) {{
                    window.opener.postMessage({{
                        type: 'oauth_success',
                        access_token: '{access_token}',
                        refresh_token: '{token.get("refresh_token", "")}',
                    }}, '*');
                    // Close the popup after a short delay
                    window.close()
                }} else {{
                    document.body.innerHTML = '<h2>Authentication Successful!</h2><p>You can close this window now.</p>';
                }}
            </script>
        </body>
        </html>
        """
        return HTMLResponse(content=html_content)

    except Exception as e:
        return HTMLResponse(content=f"""
            <!DOCTYPE html>
            <html>
            <head><title>Authentication Error</title></head>
            <body>
                <h2>Authentication Error</h2>
                <p>Failed to exchange token: {str(e)}</p>
                <p>Please close this window and try again.</p>
            </body>
            </html>
        """, status_code=500)
import hashlib
import base64
import secrets
from authlib.integrations.httpx_client import AsyncOAuth2Client

class OAuthClient:
    def __init__(self, client_id, client_secret, metadata_url):
        self.client_id = client_id
        self.client_secret = client_secret
        self.metadata_url = metadata_url
        self.client = AsyncOAuth2Client(client_id=client_id, client_secret=client_secret, scope="rancher:mcp offline_access")

    def generate_pkce_pair(self):
        """Generates code_verifier and code_challenge."""
        # 1. Create a high-entropy random string (Verifier)
        verifier = secrets.token_urlsafe(64)

        # 2. Hash it and encode it (Challenge)
        sha256_hash = hashlib.sha256(verifier.encode('utf-8')).digest()
        challenge = base64.urlsafe_b64encode(sha256_hash).decode('utf-8').replace('=', '')

        return verifier, challenge

    async def get_auth_url(self, auth_endpoint, redirect_uri):
        """Step 1: Create the URL to send the user to."""
        verifier, challenge = self.generate_pkce_pair()

        # Store 'verifier' and 'state' in a secure session or DB for Step 2
        state = secrets.token_urlsafe(16)

        url, _ = self.client.create_authorization_url(
            auth_endpoint,
            redirect_uri=redirect_uri,
            code_challenge=challenge,
            code_challenge_method='S256',
            state=state
        )
        return url, verifier, state

    async def fetch_token(self, token_endpoint, authorization_response, redirect_uri, verifier):
        """Step 2: Exchange the code for an Access Token."""
        token = await self.client.fetch_token(
            token_endpoint,
            authorization_response=authorization_response,
            redirect_uri=redirect_uri,
            code_verifier=verifier  # Crucial for PKCE
        )
        return token

    async def refresh_token(self, token_endpoint, refresh_token):
        """Refresh the access token using a refresh token."""
        token = await self.client.refresh_token(
            token_endpoint,
            refresh_token=refresh_token
        )
        return token
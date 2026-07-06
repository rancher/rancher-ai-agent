import logging
import httpx
import os
from urllib.parse import urlparse
from fastapi import Request

async def get_user_id(host: str, token: str) -> str:
    """
    Retrieves the user ID from the Rancher API using the session token.
    """
    url = f"{host}/v3/users?me=true"
    try:
        async with httpx.AsyncClient(timeout=5.0, verify=False) as client:
            resp = await client.get(url, headers={
                "Cookie": f"R_SESS={token}",
                "Accept": "application/json",
            })
            payload = resp.json()
            
            if (payload.get("type") == "error") or (resp.status_code != 200) or ("data" not in payload) or (len(payload["data"]) == 0):
                logging.error("user API returned error: %s - %s", resp.status_code, payload)
                raise Exception("Failed to retrieve user ID from Rancher API")
            
            user_id = payload["data"][0]["id"]
            
            if user_id:
                logging.info("user API returned: %s - userId %s", resp.status_code, user_id)

                return user_id
    except Exception as e:
        logging.error("user API call failed: %s", e)

    return None

async def get_user_id_from_request(request: Request) -> str:
    """
    Retrieves the user ID from the Rancher API using the session token from the request cookies.
    """
    rancher_url = os.environ.get("RANCHER_URL", "https://rancher.cattle-system")
    token = request.cookies.get("R_SESS")

    if not token:
        logging.warning("R_SESS cookie not found")
        return None

    return await get_user_id(rancher_url, token)
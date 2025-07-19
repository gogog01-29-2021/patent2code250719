import os, requests, xmltodict
CONSUMER_KEY="IAlIF82RV7dS7P9eMOyB8igK9BuqxhHsYGYoRrgmXmEBDZ25"
Consumer_Secret="7GCSocco9cGp1u9LpPJq5IpZvGWtaHrGHtSahAVnZLjuJAFF8AHxAOCo59C2VIXt"
CK = os.getenv(CONSUMER_KEY)      # set via .env or export
CS = os.getenv(Consumer_Secret)  # set via .env or export

auth = requests.auth.HTTPBasicAuth(CK, CS)

tok = requests.post(
    "https://ops.epo.org/3.2/auth/accesstoken",
    data={"grant_type": "client_credentials"},
    auth=auth, timeout=10
).json()["access_token"]

xml = requests.get(
    "https://ops.epo.org/3.2/rest-services/published-data/publication/epodoc/EP1000000/biblio",
    headers={"Authorization": f"Bearer {tok}", "Accept": "application/xml"},
    timeout=10
).text

title = xmltodict.parse(xml)["ops:world-patent-data"]["exchange-doc"]["bibliographic-data"]["invention-title"]["#text"]
print("🔖", title)

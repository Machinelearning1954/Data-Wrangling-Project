import sys
sys.path.append('/opt/.manus/.sandbox-runtime')
from data_api import ApiClient
import json

client = ApiClient()

# Fetch Apple stock data (AAPL) for the last 5 years, daily interval
stock_data = client.call_api('YahooFinance/get_stock_chart', query={'symbol': 'AAPL', 'interval': '1d', 'range': '5y', 'includeAdjustedClose': True})

with open('/home/ubuntu/data/aapl_stock_data.json', 'w') as f:
    json.dump(stock_data, f)

print('AAPL stock data saved to /home/ubuntu/data/aapl_stock_data.json')

# Fetch US GDP data (NY.GDP.MKTP.CD for United States - US)
# First, let's confirm the indicator code and country code if needed.
# The API docs for DataBank/indicator_data suggest NY.GDP.MKTP.CD and EUU (for European Union) as defaults.
# We need GDP for the US. A quick search or prior knowledge suggests 'USA' or 'US' for country code.
# Let's assume 'USA' is a valid country code for the API. If not, we might need to search for the correct one.
# For now, let's try with 'USA'. The indicator NY.GDP.MKTP.CD is 'GDP (current US$)'.

gdp_data = client.call_api('DataBank/indicator_data', query={'indicator': 'NY.GDP.MKTP.CD', 'country': 'USA'})

with open('/home/ubuntu/data/us_gdp_data.json', 'w') as f:
    json.dump(gdp_data, f)

print('US GDP data saved to /home/ubuntu/data/us_gdp_data.json')

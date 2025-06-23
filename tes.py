from recommender.recommend import recommend_vpn
from pprint import pprint

# Define user input for the VPN recommender
inputs = {
    'speed': 6.67,  # in Mbps
    'price': 5.0,   # in USD
    'trial_available': 'yes',
    'encryption': 'AES-256',
    'logging_policy': 'no_logs',
    'max_devices': 6,
    'country': 'United States'
}


# Call the recommender
recommendations = recommend_vpn(inputs)

# Display the final recommended VPNs nicely
print("\n🎯 FINAL RECOMMENDATIONS:\n")
pprint(recommendations.to_dict(orient="records"))

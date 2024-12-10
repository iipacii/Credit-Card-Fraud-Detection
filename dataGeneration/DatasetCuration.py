from openai import OpenAI
import json
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
import os
from typing import List, Dict
import logging
import uuid
import random
from geopy.distance import geodesic
from dotenv import load_dotenv
import argparse

load_dotenv()

class DiverseUserTransactionCollector:
    def __init__(self, api_key: str, output_dir: str = "transaction_data"):
        """Initialize the collector with OpenAI API key and output directory"""
        self.api_key = api_key
        self.client = OpenAI(api_key=api_key) 
        self.output_dir = output_dir
        self.setup_logging()
        self.setup_directories()
        self.progress_file = f"{output_dir}/progress.json"
        self.load_progress()
        
        # Initialize diversity tracking
        self.user_profiles = self.load_user_profiles()
        self.merchant_names = set()
        self.used_locations = []

        # Create errors directory
        os.makedirs(f"{self.output_dir}/errors", exist_ok=True)
        
        # Define geographic regions for diversity
        self.regions = [
            (40.0, 49.0, -125.0, -115.0, "Pacific Northwest"),
            (35.0, 42.0, -115.0, -105.0, "Mountain Region"),
            (30.0, 35.0, -105.0, -95.0, "South Central"),
            (35.0, 42.0, -90.0, -80.0, "Midwest"),
            (40.0, 47.0, -80.0, -70.0, "Northeast"),
            (25.0, 35.0, -85.0, -75.0, "Southeast"),
            (35.0, 42.0, -124.0, -110.0, "West Coast"),
            (25.0, 30.0, -98.0, -80.0, "Deep South")
        ]
        
        # Define expanded user archetypes with detailed characteristics
        self.user_archetypes = [
            {
                "type": "Urban Professional",
                "avg_transaction": (50, 300),
                "categories": ["Restaurant", "Transportation", "Entertainment", "Grocery", "Coffee Shops"],
                "online_ratio": 0.4,
                "peak_hours": [(8, 9), (12, 14), (18, 21)],
                "weekend_behavior": "active",
                "fraud_risk": "medium"
            },
            {
                "type": "Suburban Family",
                "avg_transaction": (70, 400),
                "categories": ["Grocery", "Retail", "Gas", "Healthcare", "Department Stores", "Home Improvement"],
                "online_ratio": 0.3,
                "peak_hours": [(9, 11), (15, 18)],
                "weekend_behavior": "very active",
                "fraud_risk": "low"
            },
            {
                "type": "Rural Resident",
                "avg_transaction": (40, 200),
                "categories": ["Gas", "Grocery", "Hardware", "Agriculture", "Automotive"],
                "online_ratio": 0.2,
                "peak_hours": [(7, 9), (16, 19)],
                "weekend_behavior": "moderate",
                "fraud_risk": "low"
            },
            {
                "type": "College Student",
                "avg_transaction": (20, 150),
                "categories": ["Restaurant", "Entertainment", "Education", "Grocery", "Books", "Coffee Shops"],
                "online_ratio": 0.5,
                "peak_hours": [(11, 23)],
                "weekend_behavior": "very active",
                "fraud_risk": "medium"
            },
            {
                "type": "Tech-Savvy Professional",
                "avg_transaction": (60, 500),
                "categories": ["Electronics", "Online Services", "Restaurant", "Entertainment", "Travel"],
                "online_ratio": 0.6,
                "peak_hours": [(10, 22)],
                "weekend_behavior": "active",
                "fraud_risk": "high"
            },
            {
                "type": "Small Business Owner",
                "avg_transaction": (100, 1000),
                "categories": ["Office Supplies", "Professional Services", "Restaurant", "Hardware", "Wholesale"],
                "online_ratio": 0.35,
                "peak_hours": [(8, 18)],
                "weekend_behavior": "moderate",
                "fraud_risk": "medium"
            },
            {
                "type": "Retiree",
                "avg_transaction": (30, 200),
                "categories": ["Healthcare", "Grocery", "Restaurant", "Retail", "Pharmacy", "Home Improvement"],
                "online_ratio": 0.25,
                "peak_hours": [(9, 16)],
                "weekend_behavior": "moderate",
                "fraud_risk": "low"
            },
            {
                "type": "Healthcare Worker",
                "avg_transaction": (40, 250),
                "categories": ["Grocery", "Restaurant", "Healthcare", "Gas", "Coffee Shops"],
                "online_ratio": 0.3,
                "peak_hours": [(6, 8), (16, 19)],
                "weekend_behavior": "varied",
                "fraud_risk": "low"
            },
            {
                "type": "Gig Economy Worker",
                "avg_transaction": (25, 200),
                "categories": ["Gas", "Restaurant", "Grocery", "Automotive", "Coffee Shops"],
                "online_ratio": 0.35,
                "peak_hours": [(7, 22)],
                "weekend_behavior": "very active",
                "fraud_risk": "medium"
            },
            {
                "type": "Luxury Shopper",
                "avg_transaction": (200, 2000),
                "categories": ["Designer Retail", "Fine Dining", "Travel", "Jewelry", "Spa and Beauty"],
                "online_ratio": 0.45,
                "peak_hours": [(11, 19)],
                "weekend_behavior": "active",
                "fraud_risk": "high"
            },
            {
                "type": "Frugal Minimalist",
                "avg_transaction": (15, 150),
                "categories": ["Grocery", "Public Transport", "Discount Stores", "Utilities"],
                "online_ratio": 0.25,
                "peak_hours": [(10, 18)],
                "weekend_behavior": "low",
                "fraud_risk": "low"
            },
            {
                "type": "Digital Nomad",
                "avg_transaction": (30, 300),
                "categories": ["Travel", "Coffee Shops", "Co-working", "Online Services", "Restaurant"],
                "online_ratio": 0.7,
                "peak_hours": [(9, 23)],
                "weekend_behavior": "active",
                "fraud_risk": "high"
            }
        ]

        # Define merchant categories with example businesses
        self.merchant_categories = {
            "Restaurant": ["Local Diner", "National Chain", "Fast Food", "Fine Dining", "Food Truck"],
            "Grocery": ["Supermarket", "Local Market", "Organic Store", "Wholesale Club"],
            "Gas": ["Gas Station", "Service Station", "Truck Stop"],
            "Retail": ["Department Store", "Clothing Store", "Sports Store", "Electronics Store"],
            "Entertainment": ["Movie Theater", "Concert Venue", "Amusement Park", "Arcade"],
            "Healthcare": ["Pharmacy", "Clinic", "Medical Lab", "Dental Office"],
            "Travel": ["Airline", "Hotel", "Car Rental", "Travel Agency"],
            "Education": ["Bookstore", "Online Course", "Tutorial Service", "School Supply"],
            "Professional Services": ["Office Supply", "Printing Service", "Consulting Firm"],
            "Home Improvement": ["Hardware Store", "Garden Center", "Home Decor"],
            "Coffee Shops": ["Local Cafe", "Chain Coffee Shop", "Bakery Cafe"],
            "Electronics": ["Computer Store", "Phone Store", "Electronics Retailer"],
            "Automotive": ["Auto Parts", "Car Wash", "Repair Shop"],
            "Online Services": ["Digital Content", "Software Service", "Cloud Storage"],
            "Designer Retail": ["Luxury Brand", "Designer Boutique", "High-end Department Store"],
            "Spa and Beauty": ["Salon", "Spa", "Beauty Supply"]
        }

        print(f"Initializing collector with output directory: {output_dir}")
        print(f"Loaded {len(self.user_archetypes)} user archetypes")
        print(f"Loaded {len(self.merchant_categories)} merchant categories")
        print(f"Configured {len(self.regions)} geographic regions")

    def setup_logging(self):
        """Set up logging configuration"""
        logging.basicConfig(
            filename='data_collection.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def setup_directories(self):
        """Create necessary directories"""
        for dir_path in [f"{self.output_dir}/{subdir}" 
                        for subdir in ['raw', 'processed', 'users']]:
            os.makedirs(dir_path, exist_ok=True)

    def load_progress(self):
        """Load progress from previous runs"""
        if os.path.exists(self.progress_file):
            with open(self.progress_file, 'r') as f:
                self.progress = json.load(f)
        else:
            self.progress = {
                'total_transactions': 0,
                'users_completed': 0,
                'last_user_id': None,
                'completed_user_ids': []
            }

    def save_progress(self):
        """Save current progress"""
        with open(self.progress_file, 'w') as f:
            json.dump(self.progress, f)

    def load_user_profiles(self) -> Dict:
        """Load existing user profiles to maintain diversity"""
        profiles = {}
        users_dir = f"{self.output_dir}/users"
        if os.path.exists(users_dir):
            for filename in os.listdir(users_dir):
                if filename.endswith('.json'):
                    file_path = f"{users_dir}/{filename}"
                    if os.path.getsize(file_path) > 0:  # Check if file is not empty
                        with open(file_path, 'r') as f:
                            try:
                                data = json.load(f)
                                profiles[data['user_profile']['user_id']] = data['user_profile']
                            except json.JSONDecodeError:
                                print(f"Error decoding JSON from file: {file_path}")
        return profiles

    def get_diverse_location(self) -> tuple:
        """Get a diverse location based on existing users"""
        used_regions = [self.get_region_index(profile['home_address']['latitude'], 
                                            profile['home_address']['longitude']) 
                       for profile in self.user_profiles.values()]
        
        # Find less used regions
        region_counts = pd.Series(used_regions).value_counts()
        all_regions = set(range(len(self.regions)))
        unused_regions = all_regions - set(region_counts.index)
        
        if unused_regions:
            region_idx = random.choice(list(unused_regions))
        else:
            region_idx = region_counts.idxmin()
            
        region = self.regions[region_idx]
        
        # Generate location within region
        lat = random.uniform(region[0], region[1])
        lon = random.uniform(region[2], region[3])
        
        return lat, lon

    def get_region_index(self, lat: float, lon: float) -> int:
        """Get the region index for a given location"""
        for i, (min_lat, max_lat, min_lon, max_lon, _) in enumerate(self.regions):
            if min_lat <= lat <= max_lat and min_lon <= lon <= max_lon:
                return i
        return -1

    def generate_user_prompt(self, user_id: str) -> str:
        """Generate diverse prompt for a single user's transactions"""
        # Select underutilized archetype
        used_archetypes = [profile.get('archetype') for profile in self.user_profiles.values()]
        archetype = random.choice([a for a in self.user_archetypes 
                                 if a['type'] not in used_archetypes[-5:]])
        
        # Get diverse location
        home_lat, home_lon = self.get_diverse_location()
        
        # Generate merchant examples for this archetype
        merchant_examples = []
        for category in archetype['categories']:
            if category in self.merchant_categories:
                merchant_examples.extend(random.sample(self.merchant_categories[category], 
                                                    min(2, len(self.merchant_categories[category]))))
        
        user_profile_str = f"""
User Profile:
- user_id: {user_id}
- archetype: {archetype['type']}
- home location: latitude {home_lat:.4f}, longitude {home_lon:.4f}
- typical transaction range: ${archetype['avg_transaction'][0]}-${archetype['avg_transaction'][1]}
- preferred categories: {', '.join(archetype['categories'])}
- online shopping ratio: {archetype['online_ratio']*100}%
- peak activity hours: {archetype['peak_hours']}
- weekend behavior: {archetype['weekend_behavior']}
- fraud risk level: {archetype['fraud_risk']}
"""

        merchant_examples_str = f"""
Sample merchants to use (mix with other realistic merchants):
{', '.join(merchant_examples)}
"""

        transaction_requirements_str = f"""
Transaction Requirements:
1. Create exactly 20 transactions within the last 30 days
2. Transaction distribution:
   - 15 transactions within 10 miles of home
   - 3-4 transactions within 10-50 miles
   - 1-2 transactions anywhere else
   - {archetype['fraud_risk']} chance of fraudulent transaction (0-2 based on risk level)

3. Follow the exact JSON structure below:

{{
    "user_profile": {{
        "user_id": "{user_id}",
        "archetype": "{archetype['type']}",
        "home_address": {{
            "latitude": {home_lat},
            "longitude": {home_lon}
        }},
        "preferred_categories": [
            "category1",
            "category2"
        ],
        "avg_transaction_amount": 100.00,
        "credit_score": 750
    }},
    "transactions": [
        {{
            "transaction_id": "TX123",
            "timestamp": "2024-11-05 10:30:00",
            "amount": 50.25,
            "merchant": {{
                "id": "M123",
                "name": "Sample Store",
                "category": "Grocery",
                "latitude": 40.7128,
                "longitude": -74.0060
            }},
            "transaction_type": "in-person",
            "location": {{
                "latitude": 40.7128,
                "longitude": -74.0060,
                "distance_from_home": 5.2  
            }},
            "device_info": {{
                "device_id": "D123",
                "type": "mobile",
                "is_trusted": true
            }},
            "authentication": {{
                "cvv_provided": true,
                "three_d_secure": "authenticated",
                "attempts": 1
            }},
            "is_fraud": false
        }}
    ]
}}
"""

        return f"""Generate 10 credit card transactions for a single user with these specific characteristics:
{user_profile_str}
{merchant_examples_str}
{transaction_requirements_str}
    Please generate 10 realistic transacti   ons following this exact structure, varying the values appropriately. Ensure all fields maintain the same format and data types as shown in the example."""

    def validate_diversity(self, new_data: Dict) -> bool:
        """Validate that new user data is sufficiently different from existing data"""
        if not self.user_profiles:  # First user is always valid
            return True
            
        new_profile = new_data['user_profile']
        new_location = (new_profile['home_address']['latitude'], 
                       new_profile['home_address']['longitude'])
        
        # Check location diversity
        for profile in self.user_profiles.values():
            existing_location = (profile['home_address']['latitude'],
                               profile['home_address']['longitude'])
            distance = geodesic(new_location, existing_location).miles
            if distance < 50:  # Too close to existing user
                return False
        
        # Check merchant diversity
        new_merchants = set(t['merchant']['name'] for t in new_data['transactions'])
        merchant_overlap = len(new_merchants.intersection(self.merchant_names)) / len(new_merchants)
        if merchant_overlap > 0.3:  # Too many similar merchants
            return False
        
        # Check transaction pattern diversity
        new_timestamps = [t['timestamp'] for t in new_data['transactions']]
        new_amounts = [t['amount'] for t in new_data['transactions']]
        
        for profile_id in self.user_profiles:
            existing_file = f"{self.output_dir}/users/{profile_id}.json"
            if os.path.exists(existing_file):
                with open(existing_file, 'r') as f:
                    existing_data = json.load(f)
                    existing_timestamps = [t['timestamp'] for t in existing_data['transactions']]
                    existing_amounts = [t['amount'] for t in existing_data['transactions']]
                    
                    # Check for too similar transaction patterns
                    timestamp_pattern = self.compare_patterns(new_timestamps, existing_timestamps)
                    amount_pattern = self.compare_patterns(new_amounts, existing_amounts)
                    
                    if timestamp_pattern > 0.7 and amount_pattern > 0.7:  # Too similar patterns
                        return False
            
        return True

    def compare_patterns(self, pattern1: List, pattern2: List) -> float:
        """Compare two patterns and return similarity score"""
        # Simple comparison - can be enhanced based on needs
        try:
            pattern1_normalized = [float(x) for x in pattern1]
            pattern2_normalized = [float(x) for x in pattern2]
            
            # Convert to numpy arrays for easier comparison
            p1 = np.array(pattern1_normalized)
            p2 = np.array(pattern2_normalized)
            
            # Normalize the patterns
            p1 = (p1 - p1.mean()) / p1.std()
            p2 = (p2 - p2.mean()) / p2.std()
            
            # Calculate correlation
            correlation = np.corrcoef(p1, p2)[0, 1]
            return abs(correlation) if not np.isnan(correlation) else 0
            
        except Exception as e:
            self.logger.warning(f"Pattern comparison error: {str(e)}")
            return 0

    async def generate_user_data(self, user_id: str) -> Dict:
        """Generate transaction data for a single user with diversity checks"""
        max_attempts = 3
        attempts = 0
        print(f"Generating data for user {user_id}")
        while attempts < max_attempts:
            
            try:
                # Updated OpenAI API call
                response = self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that generates diverse and realistic credit card transaction data. Your response must be valid JSON."},
                        {"role": "user", "content": self.generate_user_prompt(user_id)}
                    ],
                    temperature=0.8,
                    max_tokens=4000,
                )
                
                # Get the raw response content
                raw_content = response.choices[0].message.content.strip()
                
                # Log the raw response for debugging
                self.logger.info(f"Raw API response for user {user_id}:\n{raw_content}")
                
                # Try to find JSON content if it's wrapped in other text
                try:
                    # First attempt: direct JSON parsing
                    data = json.loads(raw_content)
                except json.JSONDecodeError:
                    # Second attempt: try to find JSON-like content between curly braces
                    try:
                        start_idx = raw_content.find('{')
                        end_idx = raw_content.rindex('}') + 1
                        if start_idx != -1 and end_idx != -1:
                            json_content = raw_content[start_idx:end_idx]
                            data = json.loads(json_content)
                        else:
                            raise ValueError("No JSON content found")
                    except (json.JSONDecodeError, ValueError) as e:
                        # Save the problematic response to a file for inspection
                        error_file = f"{self.output_dir}/errors/response_{user_id}_{attempts}.txt"
                        os.makedirs(os.path.dirname(error_file), exist_ok=True)
                        with open(error_file, 'w') as f:
                            f.write(raw_content)
                        self.logger.error(f"Failed to parse response. Saved to {error_file}")
                        raise e
                
                # Validate diversity
                if self.validate_diversity(data):
                    # Update tracking sets
                    self.user_profiles[user_id] = data['user_profile']
                    self.merchant_names.update(t['merchant']['name'] 
                                            for t in data['transactions'])
                    
                    # Save user data
                    with open(f"{self.output_dir}/users/{user_id}.json", 'w') as f:
                        json.dump(data, f, indent=2)
                    
                    return data
                
                attempts += 1
                self.logger.warning(f"Generated data not diverse enough, attempt {attempts}/{max_attempts}")
                
            except Exception as e:
                self.logger.error(f"Error generating data for user {user_id}: {str(e)}")
                attempts += 1
                time.sleep(2)  # Add a small delay between retries
        
        raise Exception(f"Failed to generate diverse data after {max_attempts} attempts")

    def process_user_data(self, data: Dict) -> pd.DataFrame:
        """Process single user's data into DataFrame rows"""
        processed_data = []
        user_profile = data['user_profile']
        
        for transaction in data['transactions']:
            flat_transaction = {
                'user_id': user_profile['user_id'],
                'user_archetype': user_profile['archetype'],
                'user_home_latitude': user_profile['home_address']['latitude'],
                'user_home_longitude': user_profile['home_address']['longitude'],
                'user_preferred_categories': ','.join(user_profile['preferred_categories']),
                'transaction_id': transaction['transaction_id'],
                'timestamp': transaction['timestamp'],
                'amount': transaction['amount'],
                'merchant_id': transaction['merchant']['id'],
                'merchant_name': transaction['merchant']['name'],
                'merchant_category': transaction['merchant']['category'],
                'merchant_latitude': transaction['merchant']['latitude'],
                'merchant_longitude': transaction['merchant']['longitude'],
                'transaction_type': transaction['transaction_type'],
                'transaction_latitude': transaction['location']['latitude'],
                'transaction_longitude': transaction['location']['longitude'],
                'distance_from_home': transaction['location']['distance_from_home'],
                'device_id': transaction.get('device_info', {}).get('device_id'),
                'device_type': transaction.get('device_info', {}).get('type'),
                'is_trusted_device': transaction.get('device_info', {}).get('is_trusted'),
                'cvv_provided': transaction['authentication']['cvv_provided'],
                'three_d_secure': transaction['authentication']['three_d_secure'],
                'auth_attempts': transaction['authentication']['attempts'],
                'is_fraud': transaction['is_fraud']
            }
            processed_data.append(flat_transaction)
            
        return pd.DataFrame(processed_data)

    async def collect_data(self, target_transactions: int = 1000) -> None:
        """Collect data until reaching target number of transactions"""
        try:
            all_data = []
            
            if os.path.exists(f"{self.output_dir}/processed/transactions.csv"):
                current_df = pd.read_csv(f"{self.output_dir}/processed/transactions.csv")
                all_data = current_df.to_dict('records')
                self.logger.info(f"Loaded {len(current_df)} existing transactions")
            
            while len(all_data) < target_transactions:
                user_id = str(uuid.uuid4())
                self.logger.info(f"Generating data for user {user_id}")
                
                try:
                    user_data = await self.generate_user_data(user_id)
                    user_df = self.process_user_data(user_data)
                    all_data.extend(user_df.to_dict('records'))
                    
                    # Update progress
                    self.progress['total_transactions'] = len(all_data)
                    self.progress['users_completed'] += 1
                    self.progress['last_user_id'] = user_id
                    self.progress['completed_user_ids'].append(user_id)
                    self.save_progress()
                    
                    # Save current state
                    df = pd.DataFrame(all_data)
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    df = df.sort_values(['user_id', 'timestamp'])
                    df.to_csv(f"{self.output_dir}/processed/transactions.csv", index=False)
                    df.to_parquet(f"{self.output_dir}/processed/transactions.parquet")
                    
                    self.logger.info(f"Progress: {len(all_data)}/{target_transactions} transactions")
                    
                    # Wait to avoid rate limits
                    time.sleep(1)
                    
                except Exception as e:
                    self.logger.error(f"Error processing user {user_id}: {str(e)}")
                    continue
            
            self.logger.info("Data collection completed!")
            
        except Exception as e:
            self.logger.error(f"Fatal error in data collection: {str(e)}")
            raise

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Generate credit card transaction data')
    parser.add_argument('--transactions', type=int, default=100,
                       help='Number of transactions to generate (default: 100)')
    parser.add_argument('--batch-size', type=int, default=20,
                       help='Number of transactions per user (default: 20)')
    
    args = parser.parse_args()
    
    # Load API key from .env file
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("Please set OPENAI_API_KEY in your .env file")
    
    # Calculate number of users needed
    num_users = (args.transactions + args.batch_size - 1) // args.batch_size
    
    # Initialize collector
    collector = DiverseUserTransactionCollector(api_key)
    
    print(f"Generating approximately {args.transactions} transactions using {num_users} users...")
    
    # Generate data
    import asyncio
    asyncio.run(collector.collect_data(target_transactions=args.transactions))

if __name__ == "__main__":
    main()
import yaml
with open('./config1.yaml', 'r') as file:
    prime_service = yaml.safe_load(file)

print(prime_service)

import argparse, os, time, requests, json

LAMBDA_API_KEY = os.environ["LAMBDA_API_KEY"]
SSH_KEYS = ["MBP"]
T = 5

def fetch_data(url, headers, method='get', data=None):
    return requests.post(url, headers=headers, data=json.dumps(data)).json() if method == 'post' else requests.get(url, headers=headers).json()

def check_availability(gpu_type):
    response = fetch_data('https://cloud.lambdalabs.com/api/v1/instance-types', {'Authorization': 'Basic ' + LAMBDA_API_KEY})
    for instance_type, instance_data in response['data'].items():
        if gpu_type.lower() in instance_type.lower() and instance_data['regions_with_capacity_available']:
            return instance_type, instance_data['regions_with_capacity_available'][0]["name"]
    return None, None

def create_instance(instance_type_name, region):
    response = fetch_data('https://cloud.lambdalabs.com/api/v1/instance-operations/launch', 
                        {'Authorization': 'Basic ' + LAMBDA_API_KEY, 'Content-Type': 'application/json'}, 
                        'post', 
                        {"region_name": region, "instance_type_name": instance_type_name, 
                        "ssh_key_names": SSH_KEYS, "file_system_names": [], "quantity": 1})
    return response

def main():
    parser = argparse.ArgumentParser(description='Create GPU instances.')
    parser.add_argument('--gpu', type=str, default="a100", help='The type of GPU.')
    parser.add_argument('--n', type=int, default=1, help='The number of GPUs.')
    args = parser.parse_args()
    gpu_type = f"{args.n}x_{args.gpu}"

    print(f"Looking on LambdaLabs for a machine of type: {gpu_type}, retrying every {T} seconds...")

    while True:
        instance_type_name, region = check_availability(gpu_type)
        if instance_type_name and region:
            print(f"Instance found! {instance_type_name} in {region}")
            print(json.dumps(create_instance(instance_type_name, region), indent=4))
            break
        time.sleep(T)
        
if __name__ == "__main__":
    main()

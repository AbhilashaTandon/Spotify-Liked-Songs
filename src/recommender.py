
import json


def main():
    with open('data/output.json', mode='r') as data:
        json_obj = json.load(data)


if __name__ == "__main__":
    main()

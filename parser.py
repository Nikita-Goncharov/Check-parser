import os
import json

from parser_cabala import ParserChecks

if __name__ == "__main__":
    # Code for testing
    checks_count = 0
    check_folder = "chance"  # checks directory
    checks_list = os.listdir(check_folder)

    item_log_dir = os.path.join("..", "log")
    item_debug_img_dir = os.path.join("..", "log")

    for file in checks_list:
        item_path = os.path.join(check_folder, file)
        if os.path.isfile(item_path):
            checks_count += 1
            if checks_count > 5000:
                break
            print("**********************************************")
            print(f"{str(checks_count).zfill(2)} = Current check: {item_path}")
            item_file = ParserChecks(item_path, item_log_dir, item_debug_img_dir)
            check_data, not_found_data, check_logs = item_file.get_result()
            pretty_data = json.dumps(check_data, sort_keys=True, indent=4, default=str)
            print("Not found data:", not_found_data)
            print("Check logs = ", check_logs)
            print(pretty_data)
            print("**********************************************")
    print(f"Count of checks: {checks_count}")

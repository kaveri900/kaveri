import csv

# Open the CSV file
with open("students.csv", "r") as file:
    reader = csv.DictReader(file)
    
    # Loop through each row
    for row in reader:
        if row["Status"] == "Pass":
            print(row["Name"])
